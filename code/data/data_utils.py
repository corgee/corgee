import re
import os
import copy
import logging
from itertools import islice
import time
import tqdm.autonotebook as tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, BertTokenizerFast
from tokenizers import Tokenizer as TokenizersTokenizer
from typing import Dict, Any
import hashlib
import json
from functools import reduce
from azure.datalake.store import core, lib
from cryptography.fernet import Fernet
from data.cython._read import _read_sparse_file, _read_sparse_scores_file, _read_embeddings_cpp
from main.main_utils import all_gather_reduce


logger = logging.getLogger(__name__)


def gen_password_and_encrypt(to_encode):
    passkey = Fernet.generate_key()
    fernet = Fernet(passkey)
    encrypted = fernet.encrypt(to_encode.encode())
    return passkey, encrypted


def decrypt(passkey, encrypted):
    fernet = Fernet(passkey)
    return fernet.decrypt(encrypted).decode()


def connect_adl():
    # can expose these parameters as parameters, but should be fixed for considerable future
    raise NotImplementedError
    passkey = ""
    encrypted = ""
    principal_token = lib.auth(
        tenant_id="",
        client_secret=decrypt(passkey, encrypted),
        client_id="",
    )
    return core.AzureDLFileSystem(token=principal_token, store_name="")


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def read_smat_file(file_path, num_col=None):
    with open(file_path, "rb") as fin:
        ids, data, indices, indptr = _read_sparse_file(fin)

        data = np.frombuffer(data, np.float32)
        indices = np.frombuffer(indices, np.int64)
        indptr = np.frombuffer(indptr, dtype=np.int64)

    if (num_col is not None):
        smat = sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, num_col))
    else:
        smat = sp.csr_matrix((data, indices, indptr))

    return ids, smat


def read_scores_smat_file(file_path, num_scores, num_col=None):
    with open(file_path, "rb") as fin:
        ids, data, indices, indptr = _read_sparse_scores_file(fin, num_scores)
        data = np.frombuffer(data, np.float32).reshape((-1, num_scores))
        indices = np.frombuffer(indices, np.int64)
        indptr = np.frombuffer(indptr, dtype=np.int64)

    smats = []
    for i in range(num_scores):
        if (num_col is not None):
            smat = sp.csr_matrix((data[:, i], indices, indptr), shape=(len(indptr) - 1, num_col))
        else:
            smat = sp.csr_matrix((data[:, i], indices, indptr))
        smats.append(smat.copy())

    return ids, smats


def parse_file_path(file_path):
    if 'CONCAT:' in file_path:
        dir_path, base_path = file_path.split('CONCAT:')
        return [os.path.join(dir_path, path) for path in base_path.split(';')], ' '
    elif 'CONCATSEP:' in file_path:
        dir_path, base_path = file_path.split('CONCATSEP:')
        return [os.path.join(dir_path, path) for path in base_path.split(';')], " [SEP] "
    elif 'CONCATTYPES:' in file_path:
        dir_path, base_path = file_path.split('CONCATTYPES:')
        return [os.path.join(dir_path, path) for path in base_path.split(';')], "LIST"
    else:
        return [file_path], None


def parse_dl_len(_len, len_dataset, world_size):
    return_len = _len
    if isinstance(_len, str):
        assert _len[0] == 'e', "string specification of steps should begin with e"
        try:
            return_len = int(float(_len[1:])*len_dataset)
        except ValueError:
            raise Exception(f"error in parsing dl length {_len}")

    if not isinstance(return_len, int):
        raise Exception("unknown type of num_steps")

    if world_size > 1:
        return_len = all_gather_reduce(return_len, world_size, 'mean')
        return_len = np.ceil(return_len).astype(np.int32)

    return return_len


def read_text_file(input_file_path):
    file_paths, type_separator = parse_file_path(input_file_path)
    for file_idx, file_path in enumerate(file_paths):
        logger.info(f'Reading file {file_path}')
        file_lines = [x.split('\t', 1) for x in open(file_path).readlines()]
        if file_idx == 0:
            ids = np.array([int(x[0].strip()) for x in file_lines])
            if type_separator == 'LIST':
                texts = [[x[1].strip(), *['' for _ in range(len(file_paths)-1)]] for x in file_lines]
            else:
                texts = [x[1].strip() for x in file_lines]
            if len(file_paths) > 1:
                id_index_map = get_inv(ids)
        else:
            unmapped_count = 0
            for split_line in file_lines:
                mapped_idx = id_index_map.get(int(split_line[0].strip()), -1)
                if mapped_idx == -1:
                    unmapped_count += 1
                    continue
                if type_separator == 'LIST':
                    texts[mapped_idx][file_idx] = split_line[1].strip()
                else:
                    texts[mapped_idx] += type_separator + split_line[1].strip()
            if unmapped_count > 0:
                logger.info(f'Unmapped id count {unmapped_count}')
    return ids, {'type': ('list' if type_separator == 'LIST' else 'str'), 'data': texts}


def read_embs_file(file_path, ids_file_out, embs_file_out, dim):
    _read_embeddings_cpp(file_path, ids_file_out, embs_file_out, dim)


def get_inv(arr):
    res = {}
    for i, x in enumerate(arr):
        res[x] = i
    return res


def get_inv_ids(arr):
    res = np.zeros(arr.max()+1, dtype=np.int32)
    for i, x in enumerate(arr):
        res[x] = i
    return res


def get_inv_ids_dict(arr):
    return {x: i for i, x in enumerate(arr)}


def np_load_retry(path, mmap_mode=None, num_retries=3, wait=10):
    retry_count = 0
    while retry_count < num_retries:
        try:
            return np.load(path, mmap_mode=mmap_mode)
        except OSError:
            retry_count += 1
            logger.info(f'Retrying {retry_count} time')
            time.sleep(wait)
    raise Exception(f'Failed to load {path} after {num_retries} retries')


class csrMemmap:
    """
    Load and save memory mapped csr matrix where indptr is stored in memory; indices and data are stored on disk
    """
    def __init__(self, fname):
        self.fname = fname
        self.shape = np_load_retry(self.fname+'.shape.npy')
        self.indptr = np_load_retry(self.fname+'.indptr.npy')
        self.indices = np_load_retry(self.fname+'.indices.npy', mmap_mode='r')
        self.data = np_load_retry(self.fname+'.data.npy', mmap_mode='r')

    def __getitem__(self, index):
        start, end, length = self.indptr[index], self.indptr[index+1], self.indptr[index+1]-self.indptr[index]
        if isinstance(index, np.ndarray):
            indptr = np.insert(np.cumsum(length), 0, 0)
            indices = np.concatenate([self.indices[s:e] for s, e in zip(start, end)])
            data = np.concatenate([self.data[s:e] for s, e in zip(start, end)])
        else:
            indptr = np.array([0, length], dtype=np.int32)
            indices = self.indices[start:end]
            data = self.data[start:end]
        return sp.csr_matrix((data, indices, indptr), shape=(indptr.shape[0]-1, self.shape[1]))

    @staticmethod
    def verify_path(fname):
        assert os.path.exists(fname+'.shape.npy')
        assert os.path.exists(fname+'.indptr.npy')
        assert os.path.exists(fname+'.indices.npy')
        assert os.path.exists(fname+'.data.npy')

    @staticmethod
    def dump(mat, fname):
        assert isinstance(mat, sp.csr_matrix), "Can only dump csr matrix"
        np.save(fname+'.shape.npy', mat.shape)
        np.save(fname+'.indptr.npy', mat.indptr)
        np.save(fname+'.indices.npy', mat.indices)
        np.save(fname+'.data.npy', mat.data)

    @staticmethod
    def load(fname):
        shape = np_load_retry(fname+'.shape.npy')
        indptr = np_load_retry(fname+'.indptr.npy')
        indices = np_load_retry(fname+'.indices.npy')
        data = np_load_retry(fname+'.data.npy')
        return sp.csr_matrix((data, indices, indptr), shape=(shape[0], shape[1]))


def parse_target(target):
    """ target types:
        - 'sum' -> also has 'terms' = list of dicts with {factor, lossname, lefttarget, righttarget}
        - 'dict' -> also has 'contents' = actual dict
    """
    if isinstance(target, str):
        # suitable for non-dict loss functions, format example: offer->text+0.5*offerimage->text
        # here offer, text and offerimage are encoder names
        # (factor, Optional[lossname], leftfeat, rightfeat)
        terms = []
        for lossterm in target.split('+'):
            factor = 1.0
            lossname = None
            if '*' in lossterm:
                factor, lossterm = lossterm.split('*')
                factor = float(factor)
            if ':' in lossterm:
                lossname, lossterm = lossterm.split(':')
            lefttarget, righttarget = lossterm.split('->')
            terms.append({
                'factor': factor,
                'lossname': lossname,
                'lefttarget': lefttarget,
                'righttarget': righttarget
            })
        res = {'type': 'sum', 'terms': terms}
    elif isinstance(target, dict):
        # suitable for dict loss functions, key is the name of evaluation metric, value is the indivdual target
        # format example: losstrainnce:offer->text+0.5*losstrainnce:offer->text
        # here losstrainnce is the name of loss to be used and individual target are passed in the same way as above
        res = {'type': 'dict'}
        res['contents'] = {}
        for key, value in target.items():
            res['contents'][key] = parse_target(value)
    else:
        raise NotImplementedError
    return res


def populate_targets(targets_dict, xids, yids):
    if targets_dict['type'] == 'sum':
        for term in targets_dict['terms']:
            if '~~' in term['righttarget']:
                term['righttarget'], id_type = term['righttarget'].split('~~')
                if id_type == 'x':
                    term['rightids'] = xids
                elif id_type == 'y':
                    term['rightids'] = yids
                else:
                    raise Exception('id_type should be x or y')
    elif targets_dict['type'] == 'dict':
        for targets_dict_value in targets_dict['contents'].values():
            populate_targets(targets_dict_value, xids, yids)
    else:
        raise NotImplementedError


def check_done(out_dir):
    if not check_path_in_dir(out_dir, 'done.txt'):
        raise Exception(f'Processed features not found at {out_dir}')


def check_path_in_dir(dir, path):
    return os.path.exists(os.path.join(dir, path))


def check_tokens_format(out_dir):
    if check_path_in_dir(out_dir, 'input_ids_uint16.npy'):
        return 'uint16'
    elif check_path_in_dir(out_dir, 'input_ids_uint32.npy'):
        return 'uint32'
    elif check_path_in_dir(out_dir, 'input_ids.npy') and check_path_in_dir(out_dir, 'attention_mask.npy'):
        return 'int64'
    else:
        raise Exception('Features not int64 or uint16')


def tokenize_single_process(inputs, disable_tqdm=False):
    corpus, tokenizer, bsz, process_num = inputs
    encoded_dict = {'input_ids': [], 'attention_mask': []}
    for ctr in tqdm.tqdm(range(0, len(corpus), bsz), desc=f'process {process_num}', disable=disable_tqdm):
        _tokenized = tokenizer.batch_encode(corpus[ctr: min(ctr+bsz, len(corpus))])
        encoded_dict['input_ids'].append(_tokenized['input_ids'])
        encoded_dict['attention_mask'].append(_tokenized['attention_mask'])
    encoded_dict['input_ids'] = np.vstack(encoded_dict['input_ids'])
    encoded_dict['attention_mask'] = np.vstack(encoded_dict['attention_mask'])
    return encoded_dict


def tokenize_list_single_process(inputs):
    corpus, tokenizer, bsz, process_num = inputs
    encoded_dict = {'input_ids': [], 'attention_mask': []}
    for ctr in tqdm.tqdm(range(0, len(corpus), bsz), desc=f'process {process_num}'):
        _tokenized = tokenizer.batch_encode_listfeat(corpus[ctr: min(ctr+bsz, len(corpus))])
        encoded_dict['input_ids'].append(_tokenized['input_ids'])
        encoded_dict['attention_mask'].append(_tokenized['attention_mask'])
    encoded_dict['input_ids'] = np.vstack(encoded_dict['input_ids'])
    encoded_dict['attention_mask'] = np.vstack(encoded_dict['attention_mask'])
    return encoded_dict


def tokenize_text(corpus, tokenizer, bsz=10000, num_processes=None):
    if corpus['type'] == 'list':
        tokenize_fn = tokenize_single_process
    elif corpus['type'] == 'str':
        tokenize_fn = tokenize_single_process
    else:
        raise NotImplementedError

    corpus = corpus['data']
    if num_processes is None:
        num_processes = mp.cpu_count()//2

    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return list(lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    splits = split_list(corpus, num_processes)
    splits = [(split, copy.deepcopy(tokenizer), bsz, num) for (num, split) in enumerate(splits)]

    with ProcessPoolExecutor(num_processes) as executor:
        results = list(executor.map(tokenize_fn, splits))

    logger.info('Text tokenizer: Processes joined.')
    keyslist = results[0].keys()
    encoded_dict = {k: [] for k in keyslist}
    for k in keyslist:
        encoded_dict[k] = np.vstack([result[k] for result in results])
    return encoded_dict


def prepare_text(fname, tokenizer):
    out_dir = os.path.join(fname+'.processed', tokenizer.name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'done.txt')):
        return np_load_retry(os.path.join(out_dir, 'ids.npy'), mmap_mode='r').shape[0]
    ids, text = read_text_file(fname)
    ids = ids.astype(np.int32)
    encoded_dict = tokenize_text(text, tokenizer)
    # specific optimization
    length = encoded_dict['input_ids'].shape[0]
    order = np.arange(length)
    if 'input_ids' in encoded_dict and 'attention_mask' in encoded_dict:
        # sorting causing lot of complications
        order = np.argsort((encoded_dict['input_ids']>0).sum(1))[::-1]
        if encoded_dict['input_ids'].max() < 2**16 and encoded_dict['attention_mask'].max() <= 1:
            encoded_dict['input_ids_uint16'] = encoded_dict['input_ids'].astype(np.uint16)
            del encoded_dict['attention_mask']
            del encoded_dict['input_ids']
        elif encoded_dict['input_ids'].max() < 2**32 and encoded_dict['attention_mask'].max() <= 1:
            encoded_dict['input_ids_uint32'] = encoded_dict['input_ids'].astype(np.uint32)
            del encoded_dict['attention_mask']
            del encoded_dict['input_ids']
    np.save(os.path.join(out_dir, 'ids.npy'), ids[order])
    if 'input_ids_uint16' in encoded_dict:
        np.save(os.path.join(out_dir, 'lens.npy'), (encoded_dict['input_ids_uint16'][order]>0).sum(1))
    for k, v in encoded_dict.items():
        np.save(os.path.join(out_dir, f'{k}.npy'), v[order])
    with open(os.path.join(out_dir, 'done.txt'), 'w') as fout:
        fout.write('DONE')
    return length


def prepare_single_feature(config, base_dir):
    if config['type'] == 'text':
        if config.get('no_pretok', False):
            return
        tokenizer = Tokenizer(config['tokenizer'])
        fname = os.path.join(base_dir, config['fname'])
        res = prepare_text(fname, tokenizer)
        logger.info(f"text features prepared: {fname}")
        return res
    elif config['type'] == 'embs':
        fname = os.path.join(base_dir, config['fname'])
        dim = config['dim']
        res = prepare_embs(fname, dim)
        logger.info(f"embs features prepared: {fname}")
        return res
    else:
        raise NotImplementedError


def prepare_features(config, base_dir):
    if isinstance(config, list):
        for _conf in config:
            prepare_features(_conf, base_dir)
    elif isinstance(config, str):
        pass
    elif isinstance(config, dict):
        if 'type' not in config:
            for _conf in config.values():
                prepare_features(_conf, base_dir)
        else:
            prepare_single_feature(config, base_dir)
    else:
        raise NotImplementedError


def prepare_embs(fname, dim):
    out_dir = os.path.join(fname+'.processed')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'done.txt')):
        return np_load_retry(os.path.join(out_dir, 'ids.npy'), mmap_mode='r').shape[0]
    read_embs_file(fname, os.path.join(out_dir, 'ids.npy'), os.path.join(out_dir, 'embs.npy'), dim)
    with open(os.path.join(out_dir, 'done.txt'), 'w') as fout:
        fout.write('DONE')


def prepare_vanilla_spmat(fname, numy=None):
    # prepare_scores_spmat(fname, 1, numy)  # TODO: Move to this
    out_dir = fname+'.processed'
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'done.txt')):
        return
    ids, X_Y = read_smat_file(fname, num_col=numy)
    ids = np.array(ids).astype(np.int32)
    X_Y.sort_indices()
    X_Y.sum_duplicates()
    np.save(os.path.join(out_dir, 'ids.npy'), ids)
    csrMemmap.dump(X_Y, os.path.join(out_dir, 'smat'))
    with open(os.path.join(out_dir, 'done.txt'), 'w') as fout:
        fout.write('DONE')


def prepare_scores_spmat(fname, num_scores, numy=None):
    out_dir = fname+'.processed'
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'done.txt')):
        return
    ids, X_Y_list = read_scores_smat_file(fname, num_scores, num_col=numy)
    ids = np.array(ids).astype(np.int32)
    np.save(os.path.join(out_dir, 'ids.npy'), ids)
    for mat_num, X_Y in enumerate(X_Y_list):
        X_Y.sort_indices()
        X_Y.sum_duplicates()
        csrMemmap.dump(X_Y, os.path.join(out_dir, f'smat{mat_num}'))
    with open(os.path.join(out_dir, 'done.txt'), 'w') as fout:
        fout.write('DONE')


def get_score_filter_mask(filter, s):
    exp = filter['expression']
    if isinstance(exp, int):
        score = s[exp]
    elif isinstance(exp, str):
        if exp.startswith('div(') and exp.endswith(')'):
            a, b = exp[len('div('):-1].split(',')
            a, b = int(a), int(b)
            score = s[a]/(s[b]+1e-5)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    mask = np.full((len(score)), True)
    if 'min' in filter:
        mask = np.logical_and(mask, score >= filter['min'])
    if 'max' in filter:
        mask = np.logical_and(mask, score <= filter['max'])
    if 'filter_low_frac' in filter or 'filter_high_frac' in filter:
        order = np.argsort(score)
        if 'filter_low_frac' in filter:
            num_filter = int(filter['filter_low_frac']*len(score))
            if num_filter:
                mask[order[:num_filter]] = False
        if 'filter_high_frac' in filter:
            num_filter = int(filter['filter_high_frac']*len(score))
            if num_filter:
                mask[order[-num_filter:]] = False
    return mask


def prepare_spmat(fname, numy=None, preprocess=None):
    prepare_vanilla_spmat(fname, numy)
    if preprocess is None:
        return
    out_dir = os.path.join(fname+'.processed', dict_hash(preprocess))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'done.txt')):
        return
    X_Y = csrMemmap.load(os.path.join(fname+'.processed', 'smat'))
    ids = np_load_retry(os.path.join(fname+'.processed', 'ids.npy'))
    logger.info(f'Before preprocess: rows {len(ids)} nnz {X_Y.nnz}')

    if 'score_filter' in preprocess:
        X_Y.sort_indices()
        mask = np.full((X_Y.nnz), True)
        for score_filter in preprocess['score_filter']:
            scores_file = os.path.join(os.path.dirname(fname), score_filter['file'])
            prepare_scores_spmat(scores_file, score_filter['num_scores'])
            ids_filter = np_load_retry(os.path.join(scores_file+'.processed', 'ids.npy'))
            scores_filter = []
            inv_ids_filter = get_inv_ids_dict(ids_filter)
            order = vmap_idx(ids, inv_ids_filter)
            for i in range(score_filter['num_scores']):
                spmat_filter = csrMemmap.load(os.path.join(scores_file+'.processed', f'smat{i}'))[order].tocsr()
                spmat_filter.sort_indices()
                assert (spmat_filter.indices == X_Y.indices).all()
                scores_filter.append(spmat_filter.data)
            for filter in score_filter['filters']:
                mask = np.logical_and(mask, get_score_filter_mask(filter, scores_filter))
        X_Y.data *= mask
        X_Y.eliminate_zeros()
        valid_idx = (np.array(X_Y.astype(bool).sum(1))[:, 0] > 0)
        ids, X_Y = ids[valid_idx], X_Y[valid_idx]
        logger.info(f'After filtering scores by {score_filter["file"]} : rows {len(ids)} nnz {X_Y.nnz}')

    if any([x in preprocess for x in ['min_x_per_y', 'max_x_per_y', 'filter_head_y_frac', 'filter_tail_y_frac']]):
        counts_per_y = np.array(X_Y.astype(bool).sum(0))[0]
        order = np.argsort(counts_per_y)
        filter_y = np.full((X_Y.shape[1]), False)
        if 'min_x_per_y' in preprocess:
            filter_y = np.logical_or(filter_y, counts_per_y < preprocess['min_x_per_y'])
        if 'max_x_per_y' in preprocess:
            filter_y = np.logical_or(filter_y, counts_per_y > preprocess['max_x_per_y'])
        if 'filter_head_y_frac' in preprocess:
            num_filter = int(X_Y.shape[1]*float(preprocess['filter_head_y_frac']))
            if num_filter:
                filter_y[order[-num_filter:]] = True
        if 'filter_tail_y_frac' in preprocess:
            num_filter = int(X_Y.shape[1]*float(preprocess['filter_tail_y_frac']))
            if num_filter:
                filter_y[order[:num_filter]] = True
        X_Y.data *= (~filter_y)[X_Y.indices]
        X_Y.eliminate_zeros()
        valid_idx = (np.array(X_Y.astype(bool).sum(1))[:, 0] > 0)
        ids, X_Y = ids[valid_idx], X_Y[valid_idx]
        logger.info(f'After filtering Y: rows {len(ids)} nnz {X_Y.nnz}')

    if any([x in preprocess for x in ['min_y_per_x', 'max_y_per_x', 'filter_head_x_frac', 'filter_tail_x_frac']]):
        counts_per_x = np.array(X_Y.astype(bool).sum(1))[:, 0]
        order = np.argsort(counts_per_x)
        filter_x = np.full((X_Y.shape[0]), False)
        if 'min_y_per_x' in preprocess:
            filter_x = np.logical_or(filter_x, counts_per_x < preprocess['min_y_per_x'])
        if 'max_y_per_x' in preprocess:
            filter_x = np.logical_or(filter_x, counts_per_x > preprocess['max_y_per_x'])
        if 'filter_head_x_frac' in preprocess:
            num_filter = int(X_Y.shape[0]*float(preprocess['filter_head_x_frac']))
            if num_filter:
                filter_x[order[-num_filter:]] = True
        if 'filter_tail_x_frac' in preprocess:
            num_filter = int(X_Y.shape[0]*float(preprocess['filter_tail_x_frac']))
            if num_filter:
                filter_x[order[:num_filter]] = True
        ids, X_Y = ids[~filter_x], X_Y[~filter_x]
        logger.info(f'After filtering X: rows {len(ids)} nnz {X_Y.nnz}')

    if 'normalize' in preprocess:
        X_Y = normalize(X_Y, norm=preprocess['normalize'])

    np.save(os.path.join(out_dir, 'ids.npy'), ids)
    csrMemmap.dump(X_Y, os.path.join(out_dir, 'smat'))
    with open(os.path.join(out_dir, 'done.txt'), 'w') as fout:
        fout.write('DONE')


def map_idx(idx, map_dict):
    return map_dict.get(idx, -1)


vmap_idx = np.vectorize(map_idx)


def remap_indices(index, remap):
    if isinstance(remap, dict):
        return vmap_idx(index, remap)
    elif isinstance(remap, np.ndarray):
        return remap[index]
    else:
        raise NotImplementedError


class TextFeat(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer_name, missing_feat=False, minlen=None, **kwargs):
        self.fname = fname
        self.tokenizer_name = tokenizer_name
        self.missing_feat = missing_feat
        self.out_dir = os.path.join(self.fname+'.processed', self.tokenizer_name)
        self.minlen = minlen
        check_done(self.out_dir)
        self.tokens_format = check_tokens_format(self.out_dir)
        self.prefix, self.cut_one = kwargs.get('prefix', None), False
        if self.prefix is not None:
            self.prefix = np.array(list(map(int, self.prefix.split(','))), dtype=np.int32)
            if self.prefix[0] == -1:
                self.cut_one = True
                self.prefix = self.prefix[1:]
        self.suffix = kwargs.get('suffix', None)
        self.round_tokens = kwargs.get('round_tokens', None)
        self.padding_idx = kwargs.get('padding_idx', 0)
        logger.info(f'padding idx {self.padding_idx}')

    def load_txt(self):
        self.ids = np_load_retry(os.path.join(self.out_dir, 'ids.npy'))
        if self.tokens_format == 'int64':
            self.input_ids = np_load_retry(os.path.join(self.out_dir, 'input_ids.npy'), mmap_mode='r')
            self.attention_mask = np_load_retry(os.path.join(self.out_dir, 'attention_mask.npy'), mmap_mode='r')
        elif self.tokens_format == 'uint16':
            self.input_ids_uint16 = np_load_retry(os.path.join(self.out_dir, 'input_ids_uint16.npy'), mmap_mode='r')
        elif self.tokens_format == 'uint32':
            self.input_ids_uint32 = np_load_retry(os.path.join(self.out_dir, 'input_ids_uint32.npy'), mmap_mode='r')
        else:
            raise Exception(f'format not understood {self.tokens_format}')

    def load_data(self):
        self.load_txt()
        if self.missing_feat:
            self.inv_ids = get_inv_ids_dict(self.ids)
        else:
            self.inv_ids = get_inv_ids(self.ids)
        if self.minlen is not None:
            self.valididx = self.get_valididx()

    def get_ids(self):
        if hasattr(self, 'ids'):
            ids = self.ids
        else:
            ids = np_load_retry(os.path.join(self.out_dir, 'ids.npy'))

        if self.minlen is None:
            return ids
        else:
            return ids[self.get_valididx()]

    def get_valididx(self):
        return np.where(np_load_retry(os.path.join(self.out_dir, 'lens.npy')) > self.minlen)[0]

    def __len__(self):
        return len(self.get_ids())

    def pad_prefix(self, input_ids, attention_mask, prefix, cut_one):
        num_tok = input_ids.shape[-1]
        if cut_one:
            input_ids, attention_mask = input_ids[..., 1:], attention_mask[..., 1:]
        pad_width = [(0, 0)] * (input_ids.ndim - 1) + [(len(prefix), 0)]
        input_ids = np.pad(input_ids, pad_width, 'constant', constant_values=1)[..., :num_tok]
        input_ids[..., :len(prefix)] = prefix
        attention_mask = np.pad(attention_mask, pad_width, 'constant', constant_values=1)[..., :num_tok]
        return input_ids, attention_mask

    def pad_suffix(self, input_ids, attention_mask, suffix):
        raise NotImplementedError

    def get_int64_input_ids_attention_mask(self, idx, padding_idx=0):
        if self.tokens_format == 'int64':
            _input_ids, _attention_mask = self.input_ids[idx], self.attention_mask[idx]
        elif self.tokens_format == 'uint16':
            input_ids = self.input_ids_uint16[idx]
            _input_ids, _attention_mask = input_ids.astype(np.int64), (input_ids != padding_idx).astype(np.int64)
        elif self.tokens_format == 'uint32':
            input_ids = self.input_ids_uint32[idx]
            _input_ids, _attention_mask = input_ids.astype(np.int64), (input_ids != padding_idx).astype(np.int64)
        if self.prefix is not None:
            _input_ids, _attention_mask = self.pad_prefix(_input_ids, _attention_mask, self.prefix, self.cut_one)
        if self.suffix is not None:
            _input_ids, _attention_mask = self.pad_suffix(_input_ids, _attention_mask, self.suffix)
        if self.round_tokens:
            def next_multiple(num, multiple):
                num += multiple-1
                num -= num%multiple
                return num
            max_len = next_multiple(_attention_mask.sum(-1).max(), self.round_tokens)
            _input_ids, _attention_mask = _input_ids[..., :max_len], _attention_mask[..., :max_len]
        return _input_ids, _attention_mask

    def get_fts_safe(self, index):
        if isinstance(index, int) or isinstance(index, np.int32) or isinstance(index, np.ndarray):
            mask = (index >= 0)
            input_ids = np.zeros((len(index), self.max_seq_len), dtype=np.int64)
            attention_mask = np.zeros((len(index), self.max_seq_len), dtype=np.int64)
            if np.any(mask):
                pos_index = index[mask]
                input_ids[mask], attention_mask[mask] = self.get_int64_input_ids_attention_mask(pos_index, self.padding_idx)
            maxlen = max(2, attention_mask.sum(1).max())
            input_ids = input_ids[:, :maxlen]
            attention_mask = attention_mask[:, :maxlen]
            return {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'mask': mask}
        else:
            raise NotImplementedError

    def get_fts(self, index):
        index = remap_indices(index, self.inv_ids)
        if self.missing_feat:
            return self.get_fts_safe(index)
        if isinstance(index, int) or isinstance(index, np.int32) or isinstance(index, np.ndarray):
            res = {}
            res['input_ids'], res['attention_mask'] = self.get_int64_input_ids_attention_mask(index, self.padding_idx)
            maxlen = max(2, res['attention_mask'].sum(1).max())
            res['input_ids'] = res['input_ids'][:, :maxlen]
            res['attention_mask'] = res['attention_mask'][:, :maxlen]
            return res
        else:
            raise NotImplementedError


class RawTextFeat(TextFeat):
    def __init__(self, base_dir, config, **kwargs):
        self.config = config
        self.ids, self.text = read_text_file(os.path.join(base_dir, config['fname']))
        self.text = self.text['data']
        self.inv_ids = get_inv_ids(self.ids)
    
    def load_data(self):
        return

    def get_ids(self):
        return self.ids

    def __len__(self):
        return len(self.ids)

    def get_fts(self, index):
        index = remap_indices(index, self.inv_ids)
        assert len(index.shape) == 1
        return {'text': [self.text[i] for i in index]}


class EmbFeat(torch.utils.data.Dataset):
    def __init__(self, fname, missing_feat=False):
        self.fname = fname
        self.missing_feat = missing_feat
        self.out_dir = os.path.join(self.fname+'.processed')
        check_done(self.out_dir)

    def load_emb(self):
        self.ids = np_load_retry(os.path.join(self.out_dir, 'ids.npy'))
        self.embs = np_load_retry(os.path.join(self.out_dir, 'embs.npy'), mmap_mode='r')
        self.emb_dim = self.embs.shape[1]

    def load_data(self):
        self.load_emb()
        if self.missing_feat:
            self.inv_ids = get_inv_ids_dict(self.ids)
        else:
            self.inv_ids = get_inv_ids(self.ids)

    def get_ids(self):
        if hasattr(self, 'ids'):
            return self.ids
        else:
            return np_load_retry(os.path.join(self.out_dir, 'ids.npy'))

    def __len__(self):
        return len(self.get_ids())

    def get_fts_safe(self, index):
        if isinstance(index, int) or isinstance(index, np.int32) or isinstance(index, np.ndarray):
            mask = (index >= 0)
            embs = np.zeros((len(index), self.emb_dim), dtype=self.embs.dtype)
            if np.any(mask):
                pos_index = index[mask]
                embs[mask] = self.embs[pos_index]
            return {'embs': embs,
                    'mask': mask}
        else:
            raise NotImplementedError

    def get_fts(self, index):
        index = remap_indices(index, self.inv_ids)
        if self.missing_feat:
            return self.get_fts_safe(index)
        if isinstance(index, int) or isinstance(index, np.int32) or isinstance(index, np.ndarray):
            return {'embs': self.embs[index]}
        else:
            raise NotImplementedError


class ListFeat(torch.utils.data.Dataset):
    def __init__(self, config, base_dir, **kwargs):
        self.config = config
        self.base_dir = base_dir
        self.feats = []
        self.ids = {}
        for conf_item in self.config:
            self.feats.append(create_feat(conf_item, self.base_dir, **kwargs))

    def load_data(self):
        for feat in self.feats:
            if not isinstance(feat, str):
                feat.load_data()

    def get_ids(self, mode='union'):
        """Get union of ids of all features in the list"""
        if mode not in self.ids:
            if mode == 'union':
                self.ids[mode] = reduce(np.union1d, [feat.get_ids() for feat in self.feats])
            elif mode == 'intersection':
                self.ids[mode] = reduce(np.intersect1d, [feat.get_ids() for feat in self.feats])
            else:
                raise NotImplementedError
        return self.ids[mode]

    def __len__(self):
        return len(self.get_ids())

    def get_fts(self, index, precomputed_feats={}):
        feats = []
        for feat in self.feats:
            if isinstance(feat, str):
                feats.append(precomputed_feats[feat])
            elif isinstance(feat, ListFeat) or isinstance(feat, DictFeat):
                feats.append(feat.get_fts(index, precomputed_feats))
            else:
                feats.append(feat.get_fts(index))
        return feats


class DictFeat(torch.utils.data.Dataset):
    def __init__(self, config, base_dir, **kwargs):
        self.config = config
        self.base_dir = base_dir
        self.feats = {}
        self.ids = {}
        for key, conf_item in self.config.items():
            self.feats[key] = create_feat(conf_item, self.base_dir, **kwargs)

    def load_data(self):
        for feat in self.feats.values():
            if not isinstance(feat, str):
                feat.load_data()

    def get_ids(self, mode='union'):
        """Get union of ids of all features in the dict"""
        if mode not in self.ids:
            if mode == 'union':
                self.ids[mode] = reduce(np.union1d, [feat.get_ids() for feat in self.feats.values()])
            elif mode == 'intersection':
                self.ids[mode] = reduce(np.intersect1d, [feat.get_ids() for feat in self.feats.values()])
            else:
                raise NotImplementedError
        return self.ids[mode]

    def __len__(self):
        return len(self.get_ids())

    def get_fts(self, index, precomputed_feats={}):
        feats = {}
        for key, feat in self.feats.items():
            if isinstance(feat, str):
                feats[key] = precomputed_feats[feat]
            elif isinstance(feat, ListFeat) or isinstance(feat, DictFeat):
                feats[key] = feat.get_fts(index, precomputed_feats)
            else:
                feats[key] = feat.get_fts(index)
                precomputed_feats[key] = feats[key]
        return feats


class Tokenizer():
    def __init__(self, config):
        self.config = config
        if isinstance(config, dict):
            self.type = config['type']
            self.prefix = config.get('prefix', '')
            self.max_seq_len = config.get('max_seq_len', 32)
            self.add_cls_token = config.get('add_cls_token', False)
            if self.add_cls_token:
                self.cls_token_id = config.get('cls_token_id', 1)
            # can use this for different types space for queries and offers
            self.token_type_start_id = config.get('token_type_start_id', 1)
            if self.type == 'bertfast_vocab':
                self.tokenizer = BertTokenizerFast(config['vocab_file'])
                self.name = config.get('name', config['vocab_file'].replace('/', '_'))
                self.add_special_tokens = config.get('add_special_tokens', False)
            elif self.type == 'autotokenizer':
                self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
                self.name = config.get('name', config['tokenizer_path'].replace('/', '_'))
                self.add_special_tokens = config.get('add_special_tokens', False)
            elif self.type == 'tokenizers':
                def limit_tokjson(tokjson, maxlen):
                    if tokjson['model']['type'] == 'WordPiece':
                        tokjson['model']['vocab'] = {k: v for k, v in tokjson['model']['vocab'].items() if v < maxlen}
                    elif tokjson['model']['type'] == 'BPE':
                        orig_len = len(tokjson['model']['vocab'].items())
                        if orig_len <= maxlen:
                            return tokjson
                        tokjson['model']['vocab'] = {k: v for k, v in tokjson['model']['vocab'].items() if v < maxlen}
                        tokjson['model']['merges'] = tokjson['model']['merges'][:-(orig_len-maxlen)]
                    return tokjson
                tokjson = json.loads(open(config['json_file']).read())
                max_vocab = config.get('max_vocab', None)
                self.name = config.get('name', config['json_file'].replace('/', '_'))
                if max_vocab is not None:
                    tokjson = limit_tokjson(tokjson, max_vocab)
                    self.name += str(max_vocab)
                tmp_tok_file = '__tmp_tokfile__.json'
                with open(tmp_tok_file, 'w') as fout:
                    fout.write(json.dumps(tokjson))
                self.tokenizer = TokenizersTokenizer.from_file(tmp_tok_file)
                if self.add_cls_token:
                    self.tokenizer.enable_truncation(self.max_seq_len-1)
                else:
                    self.tokenizer.enable_truncation(self.max_seq_len)
                self.tokenizer.enable_padding()
            if self.token_type_start_id > 1:
                self.name = os.path.join(f'tokenstart{self.token_type_start_id}', self.name)
            self.name = os.path.join(f'seqlen{self.max_seq_len}', self.name)
        else:
            raise NotImplementedError

    def batch_encode(self, text):
        if self.prefix != '':
            text = [self.prefix+x for x in text]
        if self.type == 'bertfast_vocab' or self.type == 'autotokenizer':
            encodings = self.tokenizer.batch_encode_plus(
                    text,  # Sentence to encode.
                    add_special_tokens=self.add_special_tokens,              # Add '[CLS]' and '[SEP]'
                    max_length=self.max_seq_len-1 if self.add_cls_token else self.max_seq_len,
                    padding='max_length',
                    return_attention_mask=True,           # Construct attn. masks.
                    return_tensors='np',                  # Return numpy tensors.
                    truncation=True
                )
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
        else:
            encodings = self.tokenizer.encode_batch(text, add_special_tokens=False)
            input_ids = np.stack([enc.ids for enc in encodings]).astype(np.int64)
            attention_mask = np.stack([enc.attention_mask for enc in encodings]).astype(np.int64)
        if self.add_cls_token:
            input_ids = np.c_[np.zeros(input_ids.shape[0], dtype=int)+self.cls_token_id, input_ids]
            attention_mask = np.c_[np.ones(attention_mask.shape[0], dtype=int), attention_mask]
        # pad if needed (for custom tokenizer case, it returns smaller ones sometimes)
        pad_len = self.max_seq_len-input_ids.shape[1]
        if pad_len > 0:
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), 'constant')
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), 'constant')
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def batch_encode_listfeat(self, textlist):
        assert self.add_special_tokens is False
        if self.type == 'bertfast_vocab' or self.type == 'autotokenizer':
            num_types, last_type_id = len(textlist[0]), self.token_type_start_id + len(textlist[0]) - 1
            for i in range(num_types):
                t_batch = [' '.join(t[:i+1]) for t in textlist]
                if self.prefix != '':
                    t_batch = [self.prefix+x for x in t_batch]
                encoding = self.tokenizer.batch_encode_plus(
                    t_batch,  # Sentence to encode.
                    add_special_tokens=False,              # Add '[CLS]' and '[SEP]'
                    max_length=self.max_seq_len-1 if self.add_cls_token else self.max_seq_len,
                    padding='max_length',
                    return_attention_mask=True,           # Construct attn. masks.
                    return_tensors='np',
                    truncation=True
                )
                if i == 0:
                    type_ids = -encoding['attention_mask'].astype(np.int64)
                elif i < (num_types - 1):
                    type_ids -= encoding['attention_mask'].astype(np.int64)
                else:
                    type_ids += last_type_id * encoding['attention_mask'].astype(np.int64)
                    input_ids = encoding['input_ids'].astype(np.int64)
        else:
            raise NotImplementedError
        if self.add_cls_token:
            input_ids = np.c_[np.zeros(input_ids.shape[0], dtype=int)+self.cls_token_id, input_ids]
            type_ids = np.c_[np.ones(type_ids.shape[0], dtype=int), type_ids]
        # pad if needed (for custom tokenizer case, it returns smaller ones sometimes)
        pad_len = self.max_seq_len-input_ids.shape[1]
        if pad_len > 0:
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), 'constant')
            type_ids = np.pad(type_ids, ((0, 0), (0, pad_len)), 'constant')
        # reusing name as attention_mask as it has support
        return {'input_ids': input_ids, 'attention_mask': type_ids}

    @staticmethod
    def get_name(config):
        if isinstance(config, dict):
            type_ = config['type']
            max_seq_len = config.get('max_seq_len', 32)
            token_type_start_id = config.get('token_type_start_id', 1)
            if type_ == 'bertfast_vocab':
                name = config.get('name', config['vocab_file'].replace('/', '_'))
            elif type_ == 'autotokenizer':
                name = config.get('name', config['tokenizer_path'].replace('/', '_'))
            elif type_ == 'tokenizers':
                max_vocab = config.get('max_vocab', None)
                name = config.get('name', config['json_file'].replace('/', '_'))
                if max_vocab is not None:
                    name += str(max_vocab)
            if token_type_start_id > 1:
                name = os.path.join(f'tokenstart{token_type_start_id}', name)
            return os.path.join(f'seqlen{max_seq_len}', name)
        else:
            raise NotImplementedError


def create_feat(config, data_dir, **kwargs):
    if isinstance(config, list):
        return ListFeat(config, data_dir, **kwargs)
    elif isinstance(config, dict):
        if 'type' not in config:  # this indicates leaf feature
            return DictFeat(config, data_dir, **kwargs)
    elif isinstance(config, str):  # indicates copy already computed feature
        return config
    else:
        raise NotImplementedError
    fname = os.path.join(data_dir, config['fname'])
    missing_feat = config.get('missing_feat', False)
    if config['type'] == 'text':
        if config.get('no_pretok', False):
            feat = RawTextFeat(data_dir, config, **kwargs)
        else:
            # avoid creating the whole tokenizer just for getting name
            tok_name = Tokenizer.get_name(config['tokenizer'])
            feat = TextFeat(fname, tok_name, missing_feat, minlen=config.get('minlen', None), **kwargs)
    elif config['type'] == 'embs':
        feat = EmbFeat(fname, missing_feat)
    else:
        raise NotImplementedError
    return feat


class XYMat():
    def __init__(self, fname, inmemory=False, preprocess=None):
        self.fname = fname
        self.inmemory = inmemory
        self.preprocess = None
        if preprocess is None:
            self.out_dir = fname+'.processed'
        else:
            self.out_dir = os.path.join(fname+'.processed', dict_hash(preprocess))
        check_done(self.out_dir)

    def load_data(self):
        self.ids = np_load_retry(os.path.join(self.out_dir, 'ids.npy'))
        if self.inmemory:
            self.smat = csrMemmap.load(os.path.join(self.out_dir, 'smat'))
        else:
            self.smat = csrMemmap(os.path.join(self.out_dir, 'smat'))
        self.shape = self.smat.shape

    def get_ids(self):
        if hasattr(self, 'ids'):
            return self.ids
        else:
            return np_load_retry(os.path.join(self.out_dir, 'ids.npy'))

    def __len__(self):
        return len(self.get_ids())
