import copy
import os
import numpy as np
import tqdm.autonotebook as tqdm
import main.xclib.xc_metrics as xc_metrics
import torch
from transformers import AutoTokenizer
from data.dl import create_dl
from data.data_utils import remap_indices, get_inv_ids_dict, csrMemmap
from main.forward_backward import Forward
from main.main_utils import all_gather_reduce, nns, check_overlap, move_to_device

from beir_utils import TOKENIZERS_MAP as BEIR_TOKENIZERS_MAP,\
    MyModelAllGPU as BEIR_MyModelAllGPU, MyModelAllGPUTokenize
from mteb import MTEB

import scipy.spatial.distance as scipy_dist
import scipy.sparse as sp
import sklearn
import pandas as pd
import logging
import json
from typing import Dict, Tuple
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)


def get_uniformity(embs):
    embs = normalize(embs)
    bsz = 512
    vals = []
    indices, num_batches = np.random.permutation(embs.shape[0]), embs.shape[0] // bsz
    for i in range(num_batches):
        vals.append(np.exp(-2*scipy_dist.pdist(embs[indices[i*bsz:(i+1)*bsz]], 'minkowski', p=2)**2).mean())
    return np.log(np.mean(vals))


def get_alignment(lembs, rembs):
    lembs, rembs = normalize(lembs), normalize(rembs)
    return (np.linalg.norm(lembs - rembs, axis=1)**2).mean()


def get_prec_rec_metrics(metrics, xy_pred=None, XY_gt=None):
    res = {}
    max_k = max(metrics['p']+metrics['r'])
    if len(metrics['p']) > 0 and len(metrics['r']) > 0:
        prec, rec = xc_metrics.precision_recall(xy_pred, XY_gt, k=max_k)
    elif len(metrics['p']) > 0:
        prec = xc_metrics.precision(xy_pred, XY_gt, k=max_k)
    elif len(metrics['r']) > 0:
        rec = xc_metrics.recall(xy_pred, XY_gt, k=max_k)
    for k in metrics['p']:
        res[f'P@{k}'] = prec[k-1]
    for k in metrics['r']:
        res[f'R@{k}'] = rec[k-1]
    return res


def get_mrr_metrics(metrics, xy_pred=None, XY_gt=None):
    res = {}
    max_k = max(metrics['mrr'])
    if len(metrics['mrr']) > 0:
        mrr = xc_metrics.mrr(xy_pred, XY_gt, k=max_k)
    for k in metrics['mrr']:
        res[f'MRR@{k}'] = mrr[k-1]
    return res

def get_ndcg_metrics(metrics, xy_pred=None, XY_gt=None):
    res = {}
    max_k = max(metrics['ndcg'])
    if len(metrics['ndcg']) > 0:
        mrr = xc_metrics.ndcg(xy_pred, XY_gt, k=max_k)
    for k in metrics['ndcg']:
        res[f'nDCG@{k}'] = mrr[k-1]
    return res

def parse_nns_metrics(metrics_str):
    valid_singleton_metrics = ["alignment", "uniformity_l", "uniformity_r", "uniformity_lr", "AUC"]
    metrics_list = list(map(lambda x: x.strip(), metrics_str.strip().split(',')))
    singleton_metrics = {singleton_metric: False for singleton_metric in valid_singleton_metrics}
    pr_metrics = {}
    for x in metrics_list:
        if x in valid_singleton_metrics:
            singleton_metrics[x] = True
        elif "@" in x:
            metrics_type, metrics_K = x.split("@")
            metrics_type, metrics_K = metrics_type.strip().lower(), int(metrics_K)
            if metrics_type not in pr_metrics:
                pr_metrics[metrics_type] = []
            pr_metrics[metrics_type].append(metrics_K)
        else:
            raise ValueError(f"invalid metric name {x}")
    for metrics_type in pr_metrics:
        pr_metrics[metrics_type] = sorted(pr_metrics[metrics_type])
    return (singleton_metrics, pr_metrics)


class NNS_Metric():
    def __init__(self, metric_conf, store, nns_config):
        self.metric_conf = metric_conf
        self.store = store
        self.nns_config = nns_config
        self.name = metric_conf['feats']
        self.singleton_metrics, self.pr_metrics = parse_nns_metrics(metric_conf['metric_names'])
        logger.info(f'parsed metrics {self.singleton_metrics} {self.pr_metrics}')

        def to_key(name):
            if name in store['keys_x']:
                return 'x'
            elif name in store['keys_y']:
                return 'y'
            else:
                raise ValueError(f"feat name {name} not found in either xfeat or yfeat")
        self.lname, self.rname = map(lambda x: x.strip(), metric_conf['feats'].split('->'))
        self.lkey, self.rkey = to_key(self.lname), to_key(self.rname)
        self.lr_key = self.lkey+self.rkey

        if self.singleton_metrics['alignment'] or self.singleton_metrics['AUC']:
            key = 'gt_'+self.lr_key+'_coo'
            if key not in store:
                self.store[key] = self.store['gt_'+self.lr_key].tocoo()

        if len(self.pr_metrics) > 0:
            self.K = max([max(x) for x in self.pr_metrics.values()])
            logger.info(f'evaluating ANNS for {self.name} with K={self.K}')
            self.filter_mat = self.store['filter_'+self.lr_key]

        if check_overlap(self.pr_metrics.keys(), ['p_dr', 'r_dr']):
            key, val_ind_key = 'gt_'+self.lr_key+'_dr', 'valids_'+self.lr_key+'_dr'
            if key not in store or val_ind_key not in store:
                mat = self.store['gt_'+self.lr_key].copy()
                mat.data = (mat.data == 0).astype(float)
                mat.eliminate_zeros()
                mat = mat.tocsr().astype(bool)
                val_ind = np.array(mat.sum(1) > 0)[:, 0]
                store[val_ind_key], store[key] = val_ind, mat[val_ind]
        if check_overlap(self.pr_metrics.keys(), ['p_ndr', 'r_ndr']):
            key, val_ind_key = 'gt_'+self.lr_key+'_ndr', 'valids_'+self.lr_key+'_ndr'
            if key not in store or val_ind_key not in store:
                mat = self.store['gt_'+self.lr_key].copy()
                mat.eliminate_zeros()
                mat = mat.tocsr().astype(bool)
                val_ind = np.array(mat.sum(1) > 0)[:, 0]
                store[val_ind_key], store[key] = val_ind, mat[val_ind]

    def __call__(self, embs) -> Tuple[Dict[str, float], Dict[str, sp.base.spmatrix]]:
        res = {}
        spmats = {}
        lembs, rembs = embs[self.lkey][self.lname], embs[self.rkey][self.rname]

        # evaluate singleton metrics
        if self.singleton_metrics['alignment']:
            lr_coo = self.store['gt_'+self.lr_key+'_coo']
            res['alignment'] = get_alignment(lembs[lr_coo.row], rembs[lr_coo.col])
        if self.singleton_metrics['uniformity_l']:
            res['uniformity_x'] = get_uniformity(lembs)
        if self.singleton_metrics['uniformity_r']:
            res['uniformity_y'] = get_uniformity(rembs)
        if self.singleton_metrics['uniformity_lr']:
            res['uniformity_xy'] = get_uniformity(np.concatenate((lembs, rembs)))
        if self.singleton_metrics['AUC']:
            lr_coo = self.store['gt_'+self.lr_key+'_coo']
            scores = (lembs[lr_coo.row]*rembs[lr_coo.col]).sum(1)
            res['AUC'] = sklearn.metrics.roc_auc_score((lr_coo.data > 0), scores)

        if len(self.pr_metrics) == 0:
            return res, spmats

        lr_pred = nns(lembs, rembs, self.nns_config, K=self.K)
        if self.filter_mat is not None:
            min_before, max_before = lr_pred.data.min(), lr_pred.data.max()
            lr_pred.data = sklearn.preprocessing.minmax_scale(lr_pred.data) + 1  # in range [1,2]
            lr_pred -= 3 * self.filter_mat
            lr_pred.data *= (lr_pred.data >= 0)
            lr_pred.eliminate_zeros()
            lr_pred.data = (lr_pred.data-1) * (max_before - min_before) + min_before
        spmats['pred_'+self.lr_key] = lr_pred

        if check_overlap(self.pr_metrics.keys(), ['p', 'r']):
            pr = get_prec_rec_metrics(
                {x: self.pr_metrics.get(x, []) for x in ['p', 'r']},
                lr_pred, self.store['gt_'+self.lr_key].astype(bool)
            )
            res.update(pr)
        if check_overlap(self.pr_metrics.keys(), ['p_dr', 'r_dr']):
            pr = get_prec_rec_metrics(
                {x.replace('_dr', ''): self.pr_metrics.get(x, []) for x in ['p_dr', 'r_dr']},
                lr_pred[self.store['valids_'+self.lr_key+'_dr']], self.store['gt_'+self.lr_key+'_dr']
            )
            res.update({'DR_'+x: v for x, v in pr.items()})
        if check_overlap(self.pr_metrics.keys(), ['p_ndr', 'r_ndr']):
            pr = get_prec_rec_metrics(
                {x.replace('_ndr', ''): self.pr_metrics.get(x, []) for x in ['p_ndr', 'r_ndr']},
                lr_pred[self.store['valids_'+self.lr_key+'_ndr']], self.store['gt_'+self.lr_key+'_ndr']
            )
            res.update({'NonDR_'+k: v for k, v in pr.items()})
        if check_overlap(self.pr_metrics.keys(), ['mrr']):
            pr = get_mrr_metrics(
                {x: self.pr_metrics.get(x, []) for x in ['mrr']},
                lr_pred, self.store['gt_'+self.lr_key].astype(bool)
            )
            res.update(pr)
        if check_overlap(self.pr_metrics.keys(), ['ndcg']):
            pr = get_ndcg_metrics(
                {x: self.pr_metrics.get(x, []) for x in ['ndcg']},
                lr_pred, self.store['gt_'+self.lr_key].astype(bool)
            )
            res.update(pr)

        return res, spmats


class NNS_Evaluator():
    def __init__(self, metrics, nns_config, model, dl, rank, world_size, device):
        self.nns_config = nns_config
        self.xembs_location = nns_config.get('xembs_location', None)
        self.yembs_location = nns_config.get('yembs_location', None)
        self.model = model
        self.dl = dl
        self.rank = rank
        self.world_size = world_size
        self.device = device
        if self.rank == 0:
            self.store = {}
            self.store['gt_xy'] = sp.csr_matrix((self.dl.xy_smat.data,
                                                 remap_indices(self.dl.xy_smat.indices,
                                                               get_inv_ids_dict(self.dl.ydl.dataset)),
                                                 self.dl.xy_smat.indptr),
                                                shape=(len(self.dl.xdl.dataset), len(self.dl.ydl.dataset)))
            self.store['gt_yx'] = self.store['gt_xy'].transpose().tocsr()
            self.store['gt_xx'] = sp.identity(len(self.dl.xdl), format='csr')
            self.store['gt_yy'] = sp.identity(len(self.dl.ydl), format='csr')
            self.store['keys_x'] = self.dl.xfeat.feats.keys()
            self.store['keys_y'] = self.dl.yfeat.feats.keys()
            if self.dl.xy_filter is not None:
                self.store['filter_xy'] = self.dl.xy_filter
                self.store['filter_yx'] = self.dl.xy_filter.transpose().tocsr()
            else:
                self.store['filter_xy'] = None
                self.store['filter_yx'] = None
            self.metrics = [NNS_Metric(metric_conf, self.store, self.nns_config) for metric_conf in metrics]

    def get_embs(self, feat_dl, desc):
        embs = []
        pbar = tqdm.tqdm(feat_dl, desc=desc, disable=(self.rank != 0))
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.model.eval()
            for feat_batch in pbar:
                emb_batch = None
                if feat_batch is not None:
                    feat_batch = move_to_device(feat_batch, self.device)
                    emb_batch = {k: v.detach().cpu().numpy() if v is not None else None
                                 for k, v in self.model(feat_batch).items()}
                if self.world_size > 1:
                    emb_batch = all_gather_reduce(emb_batch, self.world_size, 'dictitems_npvstack')
                if self.rank == 0:
                    embs.append(emb_batch)
            self.model.train()
        if self.rank == 0:
            embs = {k: np.vstack([emb[k] for emb in embs]) for k in embs[0]}
        return embs

    def __call__(self, *args, **kwargs):
        metrics, preds = {}, {}

        xembs, yembs = {}, {}
        if self.xembs_location is None:
            xembs = self.get_embs(self.dl.xdl, desc='Computing x features')
        if self.yembs_location is None:
            yembs = self.get_embs(self.dl.ydl, desc='Computing y features')

        if self.rank != 0:
            return metrics, preds

        preds.update({('embs', k): v for k, v in xembs.items()})
        preds.update({('embs', k): v for k, v in yembs.items()})
        if self.xembs_location is not None:
            keyname = os.path.basename(self.xembs_location)[:-len('.npy')]
            xembs = {keyname: np.load(self.xembs_location)}
        if self.yembs_location is not None:
            keyname = os.path.basename(self.yembs_location)[:-len('.npy')]
            yembs = {keyname: np.load(self.yembs_location)}

        if self.rank != 0:
            return metrics, preds

        for metric in self.metrics:
            metrics_, spmats_pred_ = metric({'x': xembs, 'y': yembs})
            metrics.update({(metric.name, k): v for k, v in metrics_.items()})
            preds.update({(metric.name, 'spmat_'+k): v for k, v in spmats_pred_.items()})
        return metrics, preds


class Loss_Evaluator():
    def __init__(self, config, model, rank, world_size, device):
        self.config = config
        self.model = model
        self.loss = self.model.loss
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.batch_sizes = list(map(int, config['batch_sizes'].split(','))) if 'batch_sizes' in config else None
        self.layerwise = config.get('layerwise', False)
        self.forward = Forward(config['forward'], self.model, self.rank, self.world_size, self.device)
        if self.batch_sizes is not None:
            data_config = copy.deepcopy(config['data'])
            data_config['batch_size'] = max(self.batch_sizes)
        self.dl = create_dl(data_config, rank, world_size, device, rank, world_size)

    def __call__(self, *args, **kwargs):
        self.forward.model.eval()
        losses = []
        self.dl.reset()
        for x in self.dl:
            if self.batch_sizes:
                x['target']['max_batch_sizes'] = self.batch_sizes
            losses.append(all_gather_reduce(self.forward(x, return_layerwise=self.layerwise),
                                            self.world_size, 'dictitems_mean'))
        loss = {k: np.mean([loss_[k] for loss_ in losses]) for k in losses[0]}
        self.forward.model.train()
        return loss, {}


class BEIR_Evaluator():
    def __init__(self, config, model, rank, world_size, device):
        self.config = config
        # model is not used at all
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.query_tokenizer_name = config.get('query_tokenizer_name', config.get('tokenizer_name', None))
        self.doc_tokenizer_name = config.get('doc_tokenizer_name', config.get('tokenizer_name', None))
        assert 'query_tokenizer_name' in config or 'tokenizer_name' in config
        assert 'doc_tokenizer_name' in config or 'tokenizer_name' in config

        self.beir_query_prefix = config.get('query_prefix', None)
        self.beir_doc_prefix = config.get('doc_prefix', None)
        self.beir_query_suffix = config.get('query_suffix', None)
        self.beir_doc_suffix = config.get('doc_suffix', None)
        self.tokenizer = None
        if self.beir_query_prefix is not None:
            if self.beir_query_prefix == "e5_instruct":
                self.tokenizer = config['instruct_tokenizer']
            else:
                self.beir_query_prefix = np.array(list(map(int, self.beir_query_prefix.split(','))), dtype=np.int32)
        if self.beir_doc_prefix is not None:
            self.beir_doc_prefix = np.array(list(map(int, self.beir_doc_prefix.split(','))), dtype=np.int32)
        if self.beir_query_suffix is not None:
            self.beir_query_suffix = np.array(list(map(int, self.beir_query_suffix.split(','))), dtype=np.int32)
        if self.beir_doc_suffix is not None:
            self.beir_doc_suffix = np.array(list(map(int, self.beir_doc_suffix.split(','))), dtype=np.int32)
        self.dtype = config.get('dtype', None)
        self.pq_config = config.get('pq_config', None)
        self.cache_dir = config.get('cache_dir', None)

        # if needed, at some point support different models on both sides
        self.encoder_name = config['encoder']
        self.max_per_gpu_bsz = config['max_per_gpu_bsz']
        assert self.query_tokenizer_name in BEIR_TOKENIZERS_MAP, f"tokenizer name {self.query_tokenizer_name} should be in templates"
        assert self.doc_tokenizer_name in BEIR_TOKENIZERS_MAP, f"tokenizer name {self.doc_tokenizer_name} should be in templates"
        self.tasks = config['tasks'].split(',')
        self.lang_list = ['en']
        self.BEIR_CACHE_DIR = config['BEIR_CACHE_DIR']

    def process_results(self, out_dir, tasks):
        metrics_all = {}
        cqa_dupstack_metrics = []  # ndcg10, r100, r1k
        all_metrics = []  # ndcg10, r100, r1k
        for task in tasks:
            with open(os.path.join(out_dir, task+'.json')) as f:
                metrics = json.load(f)
                test_split = 'dev' if task.lower() == 'msmarco' else 'test'
                metrics_all.update({
                    f'{task}_nDCG@10': metrics[test_split]['ndcg_at_10'],
                    f'{task}_R@100': metrics[test_split]['recall_at_100'],
                    f'{task}_R@1000': metrics[test_split]['recall_at_1000']
                })
                if task.startswith('CQADupstack'):
                    cqa_dupstack_metrics.append([metrics[test_split]['ndcg_at_10'],
                                                metrics[test_split]['recall_at_100'],
                                                metrics[test_split]['recall_at_1000']])
                else:
                    all_metrics.append([metrics[test_split]['ndcg_at_10'],
                                                metrics[test_split]['recall_at_100'],
                                                metrics[test_split]['recall_at_1000']])
        if len(cqa_dupstack_metrics) == 12:
            cqa_dupstack_metrics = np.array(cqa_dupstack_metrics).mean(0)
            metrics_all.update({
                'CQADupstack_nDCG@10': cqa_dupstack_metrics[0],
                'CQADupstack_R@100': cqa_dupstack_metrics[1],
                'CQADupstack_R@1000': cqa_dupstack_metrics[2]
            })
            all_metrics.append(cqa_dupstack_metrics.tolist())
        if len(all_metrics) == 15:
            all_metrics = np.array(all_metrics).mean(0)
            metrics_all.update({
                'BEIR_nDCG@10': all_metrics[0],
                'BEIR_R@100': all_metrics[1],
                'BEIR_R@1000': all_metrics[2]
            })
        return metrics_all

    def __call__(self, out_dir, num_batch, num_data, *args, **kwargs):
        metrics = {}
        step_out_dir = os.path.join(out_dir, f'step{num_batch}')
        if self.rank == 0:
            model_to_eval = self.model.encoders[self.encoder_name]
            if self.world_size > 1:
                model_to_eval = model_to_eval.module
            if hasattr(model_to_eval, '_orig_mod'):
                model_to_eval = model_to_eval._orig_mod
            model = BEIR_MyModelAllGPU(copy.deepcopy(model_to_eval).to('cpu'), self.max_per_gpu_bsz,
                                       query_prefix=self.beir_query_prefix,
                                       doc_prefix=self.beir_doc_prefix,
                                       query_suffix=self.beir_query_suffix,
                                       doc_suffix=self.beir_doc_suffix,
                                       dtype=self.dtype,
                                       tokenizer=self.tokenizer,
                                       tqdm=True)
            evaluation = MTEB(tasks=self.tasks,
                              beir_cache_dir=self.BEIR_CACHE_DIR,
                              task_langs=self.lang_list,
                              beir_pretok_query=self.query_tokenizer_name,
                              beir_pretok_doc=self.doc_tokenizer_name)
            evaluation.run(model, output_folder=step_out_dir,
                           pq_config=self.pq_config,
                           cache_dir=self.cache_dir)
            metrics = self.process_results(step_out_dir, self.tasks)
        return metrics, {}


class MTEB_Evaluator():
    def __init__(self, config, model, rank, world_size, device):
        self.config = config
        # model is not used at all
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.tokenizer_name = config['tokenizer_name']
        assert self.tokenizer_name in BEIR_TOKENIZERS_MAP
        self.normalize = config.get('normalize', False)
        self.scale = config.get('scale', 1.0)

        self.query_prefix = config.get('query_prefix', '')
        self.doc_prefix = config['doc_prefix']
        self.dtype = config.get('dtype', None)

        # if needed, at some point support different models on both sides
        self.encoder_name = config['encoder']
        self.max_per_gpu_bsz = config['max_per_gpu_bsz']
        self.tasks = config['tasks'].split(',')
        self.lang_list = ['en']
        self.HF_CACHE_DIR = config.get('HF_CACHE_DIR', None)
        self.cls_in_middle = config.get('cls_in_middle', False)

        self.metric = config['metric']

    def process_results(self, out_dir, tasks):
        def get_metric(metrics, name):
            if name in metrics:
                return metrics[name]
            if name == 'cos_sim_spearman' and 'cos_sim' in metrics:
                return metrics['cos_sim']['spearman']
            if name == 'cos_sim_ap' and 'cos_sim' in metrics:
                return metrics['cos_sim']['ap']
            if 'en' in metrics:
                return get_metric(metrics['en'], name)
            if 'en-en' in metrics:
                return get_metric(metrics['en-en'], name)
            import pdb; pdb.set_trace()
            raise Exception
        metrics_all = {}
        all_metrics = []
        for task in tasks:
            with open(os.path.join(out_dir, task+'.json')) as f:
                metrics = json.load(f)
                test_split = 'test'
                metrics_all[task] = get_metric(metrics[test_split], self.metric)
                all_metrics.append(get_metric(metrics[test_split], self.metric))
        metrics_all['all'] = np.array(all_metrics).mean()
        return metrics_all

    def __call__(self, out_dir, num_batch, num_data, *args, **kwargs):
        metrics = {}
        step_out_dir = os.path.join(out_dir, f'step{num_batch}')
        if self.rank == 0:
            model_to_eval = self.model.encoders[self.encoder_name]
            if self.world_size > 1:
                model_to_eval = model_to_eval.module
            if hasattr(model_to_eval, '_orig_mod'):
                model_to_eval = model_to_eval._orig_mod
            model = MyModelAllGPUTokenize(copy.deepcopy(model_to_eval).to('cpu'), self.max_per_gpu_bsz,
                                       query_prefix=self.query_prefix,
                                       doc_prefix=self.doc_prefix,
                                       tokenizer=self.tokenizer_name,
                                       dtype=self.dtype,
                                       normalize=self.normalize,
                                       scale=self.scale,
                                       cls_in_middle=self.cls_in_middle)
            evaluation = MTEB(tasks=self.tasks,
                              hf_repo_cache_dir_=self.HF_CACHE_DIR,
                              task_langs=self.lang_list)
            evaluation.run(model, output_folder=step_out_dir)
            metrics = self.process_results(step_out_dir, self.tasks)
        return metrics, {}


class Evaluator():
    def __init__(self, config, model, tb_writer, _wandb_h, mlflow_h, out_dir, rank, world_size, device, forward_backward=None):
        self.config = config
        self.model = model
        self.tb_writer = tb_writer
        self.wandb_h = _wandb_h
        self.mlflow_h = mlflow_h
        self.out_dir = out_dir
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.forward_backward = forward_backward

        if isinstance(config, list):
            self.type = 'list'
            self.evaluators = []
            for conf_ in config:
                self.evaluators.append(Evaluator(conf_, model, tb_writer, _wandb_h, mlflow_h, out_dir, rank, world_size, device, self.forward_backward))
        else:
            self.type = config['type']
            self.eval_interval = config['eval_interval']
            self.last_evaluated = 0
            self.save_pred = config.get('save_pred', False)
            self.name = config['name']

            if self.type == 'nns':
                allowed_formats = ['mm_eval']
                if config['data']['format'] not in allowed_formats:
                    raise Exception(f'ANNS evaluation requires data format in {allowed_formats}')
                self.dl = create_dl(config['data'], rank, world_size, device, rank, world_size)
                self.evaluator = NNS_Evaluator(config['metrics'], config['nns'], self.model,
                                               self.dl, self.rank, self.world_size, self.device)
            elif self.type == 'loss':
                allowed_formats = ['mm']
                if config['data']['format'] not in allowed_formats:
                    raise Exception(f'Loss evaluation requires data format in {allowed_formats}')
                self.evaluator = Loss_Evaluator(config, model, self.rank, self.world_size, self.device)

            elif self.type == 'beir':
                self.evaluator = BEIR_Evaluator(config, model, self.rank, self.world_size, self.device)

            elif self.type == 'mteb':
                self.evaluator = MTEB_Evaluator(config, model, self.rank, self.world_size, self.device)

            else:
                raise NotImplementedError

    def __call__(self, num_batch, num_data, force=False):
        if self.type == 'list':
            for evaluator in self.evaluators:
                evaluator(num_batch, num_data, force=force)
        else:
            if force or (self.last_evaluated < num_data//self.eval_interval):
                if self.forward_backward is not None:
                    logger.info('shifting fwd bwd to eval')
                    self.forward_backward.eval()
                metrics, preds = self.evaluator(self.out_dir, num_batch, num_data)
                if self.forward_backward is not None:
                    logger.info('shifting fwd bwd to train')
                    self.forward_backward.train()
                self.last_evaluated = num_data//self.eval_interval
                if self.rank == 0:
                    if self.tb_writer:
                        for key, value in metrics.items():
                            if isinstance(key, tuple):
                                key = '/'.join(list(key))
                            assert isinstance(key, str)
                            self.tb_writer.add_scalar(f'metrics/{self.name}/{key}', value, num_data)
                    if self.wandb_h is not None:
                        self.wandb_h.log({'num_data': num_data}, step=num_batch)
                        for key, value in metrics.items():
                            if isinstance(key, tuple):
                                key = '/'.join(list(key))
                            assert isinstance(key, str)
                            self.wandb_h.log({f'metrics/{self.name}/{key}': value}, step=num_batch)
                    if self.mlflow_h is not None:
                        self.mlflow_h.log('num_data', num_data, num_batch)
                        for key, value in metrics.items():
                            if isinstance(key, tuple):
                                key = '/'.join(list(key))
                            assert isinstance(key, str)
                            self.mlflow_h.log(f'metrics/{self.name}/{key}', value, num_batch)
                    if self.save_pred:
                        os.makedirs(os.path.join(self.out_dir, self.name, 'preds'), exist_ok=True)
                        for key, value in preds.items():
                            if isinstance(key, tuple):
                                key = os.path.join(*key)
                            assert isinstance(key, str)
                            if isinstance(value, sp.csr_matrix):
                                filepath = os.path.join(self.out_dir, self.name, 'preds', key+'.csrmemmap')
                                filepath = filepath.replace('->', 'to')
                                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                                csrMemmap.dump(value, filepath)
                            elif isinstance(value, np.ndarray):
                                filepath = os.path.join(self.out_dir, self.name, 'preds', key+'.npy')
                                filepath = filepath.replace('->', 'to')
                                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                                np.save(filepath, value)
                            else:
                                raise NotImplementedError
                    os.makedirs(os.path.join(self.out_dir, self.name), exist_ok=True)
                    # # TODO: remove this from comment after fixing or handling OS error here
                    # with open(os.path.join(self.out_dir, self.name, 'metrics.pkl'), 'wb') as handle:
                    #     pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Metrics for data: {self.name} \n{pd.DataFrame(metrics, index=[0])}")
