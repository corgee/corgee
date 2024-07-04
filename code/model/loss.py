import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
from model.model_helpers import GatherLayer
from model.encoders import create_encoder
from main.main_utils import split_by_max_batch_size


logger = logging.getLogger(__name__)


def create_loss(config, encoders, rank, world_size, device):
    assert isinstance(config, dict)
    if 'type' not in config:
        return DictLoss(config, encoders, rank, world_size, device)
    elif config['type'] == 'ce':
        return CELoss(config, rank, world_size, device)
    elif config['type'] == 'nce':
        return NCELoss(config, rank, world_size, device)
    elif config['type'] == 'siglip':
        return SigLIPLoss(config, rank, world_size, device)
    elif config['type'] == 'uniformity':
        return UniformityLoss(config, rank, world_size, device)
    elif config['type'] == 'alignment':
        return AlignmentLoss(config, rank, world_size, device)
    elif config['type'] == 'simsiam':
        return SimSiamLoss(config, rank, world_size, device)
    elif config['type'] == 'mae':
        return MAELoss(config, encoders, rank, world_size, device)
    elif config['type'] == 'dict_select':
        return DictSelectLoss(config, rank, world_size, device)
    else:
        raise NotImplementedError


def cross_max_sum(l, r):
    return torch.einsum('xik,yjk->xyij', l, r).max(axis=-1).values.sum(-1)


def cross_sum_sum(l, r):
    return torch.einsum('xik,yjk->xyij', l, r).sum(-1).sum(-1)


def cross_max_max(l, r):
    return torch.einsum('xik,yjk->xyij', l, r).max(axis=-1).values.max(axis=-1).values


def cross_sum_max(l, r):
    return torch.einsum('xik,yjk->xyij', l, r).sum(-1).max(axis=-1).values


cross_sim_fn_dict = {
    'maxsim_colbert': cross_max_sum,
    'cross_max_sum': cross_max_sum,
    'cross_max_max': cross_max_max,
    'cross_sum_sum': cross_sum_sum,
    'cross_sum_max': cross_sum_max
}


def multiq_max(l, r):
    return torch.einsum('xik,yk->xyi', l, r).max(axis=-1).values


def multiq_sum(l, r):
    return torch.einsum('xik,yk->xy', l, r)


def multio_max(l, r):
    return torch.einsum('xk,yjk->xyj', l, r).max(axis=-1).values


def multio_sum(l, r):
    return torch.einsum('xk,yjk->xyj', l, r).sum(axis=-1)


class AccuracyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, pos_idx):
        return (logits.argmax(axis=1)==pos_idx).sum()/pos_idx.shape[0]


class NCELoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        if 'logit_scale' in config:  # if logit scale is given, use it
            self.logit_scale = nn.Parameter(torch.tensor(config['logit_scale'], dtype=torch.float),
                                            requires_grad=config.get('train_logit_scale', False))
        else:                        # else initialize it to log(1/0.07) and train it
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sim = config.get('sim', 'cosine')
        assert self.sim in ['cosine', 'dot', 'cls_cosine', 'cls_dot', 'maxsim_colbert', \
            'cross_max_sum', 'cross_max_max', 'cross_sum_sum', 'cross_sum_max', \
            'multiq_max', 'multiq_sum', 'multio_max', 'multio_sum', \
            'cosine_64_sum_sum', 'cosine_64_max_sum', 'cosine_64_sum_max', 'cosine_64_max_max']
        self.maxbatch_sim = config.get('maxbatch_sim', None)
        self.logit_scale_max = config.get('logit_scale_max', None)
        self.all_gather = config.get('all_gather', True)
        self.both_sides = config.get('both_sides', False)
        self.ignore_score_eq_pos = config.get('ignore_score_eq_pos', False)
        self.ignore_score_gteq_pos = config.get('ignore_score_gteq_pos', False)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        self.neg = config.get('neg', None)
        self.matryoshka = config.get('matryoshka', None)  # dict of dim to weight
        if self.neg:
            assert self.neg['type'] in ['random', 'topk']
        if config.get('accuracy', False):
            self.criterion = AccuracyCriterion()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def clip_logits(self):
        if self.logit_scale_max is not None:
            self.logit_scale.data.clip_(None, self.logit_scale_max)

    def compute_sim(self, left_emb, right_emb):
        if self.sim == 'cosine':
            return F.normalize(left_emb, dim=-1) @ F.normalize(right_emb, dim=-1).T
        elif self.sim == 'dot':
            return left_emb @ right_emb.T
        elif self.sim == 'cls_cosine':
            return F.normalize(left_emb[:,0], dim=-1) @ F.normalize(right_emb[:,0], dim=-1).T
        elif self.sim.startswith('cosine_64_'):
            def agg(sims, agg):
                if agg == 'max':
                    sims = sims.max(-1).values
                elif agg == 'sum':
                    sims = sims.sum(-1)
                else:
                    raise NotImplementedError
                return sims
            def sim_fn(left_emb, right_emb, agg1, agg2):
                sims = left_emb.unsqueeze(1) @ right_emb.transpose(-1,-2)
                sims = agg(sims, agg2)
                sims = agg(sims, agg1)
                return sims
            l = F.normalize(left_emb.view((left_emb.shape[0], -1, 64)), dim=-1)
            r = F.normalize(right_emb.view((right_emb.shape[0], -1, 64)), dim=-1)
            operator = self.sim[len('cosine_64_'):]
            if self.maxbatch_sim is not None:
                res = []
                for lsplit in split_by_max_batch_size(l, self.maxbatch_sim):
                    res.append(sim_fn(lsplit, r, operator[:3], operator[4:7]))
                return torch.cat(res)
            return sim_fn(l, r, operator[:3], operator[4:7])
        elif self.sim == 'cls_dot':
            return left_emb[:,0] @ right_emb[:,0].T
        elif self.sim == 'maxsim_colbert' or self.sim.startswith('cross_'):
            cross_sim_fn = cross_sim_fn_dict[self.sim]
            if self.maxbatch_sim is not None:
                r = F.normalize(right_emb, dim=-1)
                res = []
                for l in split_by_max_batch_size(F.normalize(left_emb, dim=-1), self.maxbatch_sim):
                    res.append(cross_sim_fn(l, r))
                return torch.cat(res)
            return cross_sim_fn(F.normalize(left_emb, dim=-1), F.normalize(right_emb, dim=-1))
        elif self.sim == 'multiq_sum':
            if self.maxbatch_sim is not None:
                r = F.normalize(right_emb[:,0], dim=-1)
                res = []
                for l in split_by_max_batch_size(F.normalize(left_emb, dim=-1), self.maxbatch_sim):
                    res.append(multiq_sum(l, r))
                return torch.cat(res)
            return multiq_sum(F.normalize(left_emb, dim=-1), F.normalize(right_emb[:,0], dim=-1))
        elif self.sim == 'multiq_max':
            if self.maxbatch_sim is not None:
                r = F.normalize(right_emb[:,0], dim=-1)
                res = []
                for l in split_by_max_batch_size(F.normalize(left_emb, dim=-1), self.maxbatch_sim):
                    res.append(multiq_max(l, r))
                return torch.cat(res)
            return multiq_max(F.normalize(left_emb, dim=-1), F.normalize(right_emb[:,0], dim=-1))
        elif self.sim == 'multio_sum':
            if self.maxbatch_sim is not None:
                l = F.normalize(left_emb[:,0], dim=-1)
                res = []
                for r in split_by_max_batch_size(F.normalize(right_emb, dim=-1), self.maxbatch_sim):
                    res.append(multio_sum(l, r))
                return torch.cat(res, axis=1)
            return multio_sum(F.normalize(left_emb[:,0], dim=-1), F.normalize(right_emb, dim=-1))
        elif self.sim == 'multio_max':
            if self.maxbatch_sim is not None:
                l = F.normalize(left_emb[:,0], dim=-1)
                res = []
                for r in split_by_max_batch_size(F.normalize(right_emb, dim=-1), self.maxbatch_sim):
                    res.append(multio_max(l, r))
                return torch.cat(res, axis=1)
            return multio_max(F.normalize(left_emb[:,0], dim=-1), F.normalize(right_emb, dim=-1))

    def compute_loss(self, left_emb, right_emb, pos_idx, right_ids):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * self.compute_sim(left_emb, right_emb)
        if self.neg:
            pos_idx_2d = pos_idx.unsqueeze(1)
            # pos logits [bsz, 1]
            pos_logits = torch.gather(logits, 1, pos_idx_2d)
            # [bsz, #cand] with positives at -100
            neg_logits_all = torch.scatter(logits, 1, pos_idx_2d, torch.zeros_like(pos_idx_2d).to(logits.dtype)-100)
            # [bsz, #cand-1] all negatives sorted by logit
            negidx_sorted = torch.argsort(neg_logits_all, descending=True)[:, :-1]
            if self.neg['type'] == 'topk':
                neg_idx = negidx_sorted[:, :self.neg['num']]
            elif self.neg['type'] == 'random':
                rand_indices = torch.topk(torch.randn_like(negidx_sorted.to(torch.float)), self.neg['num']).indices
                neg_idx = torch.gather(negidx_sorted, 1, rand_indices)
            neg_logits = torch.gather(logits, 1, neg_idx)
            # [batch_size, num_negatives+1]
            logits = torch.cat((pos_logits, neg_logits), axis=1)
            pos_idx = torch.zeros_like(pos_idx)
        if right_ids is not None:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = (right_ids[pos_idx_2d] == right_ids.unsqueeze(0))
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        if self.ignore_score_eq_pos:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = (logits == torch.gather(logits, 1, pos_idx_2d))
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        if self.ignore_score_gteq_pos:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = (logits >= torch.gather(logits, 1, pos_idx_2d))
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        return self.criterion(logits, pos_idx)

    def forward_one_sided(self, left_emb, right_emb, right_ids=None):
        if self.all_gather and self.world_size > 1:
            right_emb = GatherLayer.apply(right_emb, dist.group.WORLD, self.rank)
            if right_ids is not None:
                right_ids = GatherLayer.apply(right_ids, dist.group.WORLD, self.rank)
            assert right_emb.shape[0] == left_emb.shape[0] * self.world_size
            pos_idx = torch.arange(self.rank*left_emb.shape[0], (self.rank+1)*left_emb.shape[0], device=self.device)
        else:
            pos_idx = torch.arange(left_emb.shape[0], device=self.device)
        if len(right_emb.shape) == 3:
            # flatten such that positives come first and negatives after them
            right_emb = right_emb.transpose(0,1).reshape(-1, right_emb.shape[-1])
        if self.matryoshka is None:
            return self.compute_loss(left_emb, right_emb, pos_idx, right_ids)
        else:
            lossterms_eval = []
            # 64:1, 128: 1, 256:1
            for dim, weight in self.matryoshka.items():
                lossterms_eval.append(weight*self.compute_loss(left_emb[..., :dim], right_emb[..., :dim], pos_idx, right_ids))
            return torch.stack(lossterms_eval).sum()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith('detach(') and name.endswith(')'):
                return embs[name[len('detach('):-len(')')]].detach()
            else:
                raise NotImplementedError
        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            lossterms_eval = [self.forward_one_sided(left_emb, right_emb, lossterm.get('rightids'))]
            if self.both_sides:
                lossterms_eval.append(self.forward_one_sided(right_emb, left_emb))
            loss += lossterm['factor']*torch.stack(lossterms_eval).mean()
        return loss


class SigLIPLoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        if 'logit_scale' in config:  # if logit scale is given, use it
            self.logit_scale = nn.Parameter(torch.tensor(config['logit_scale'], dtype=torch.float),
                                            requires_grad=config.get('train_logit_scale', False))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * 10)
        if 'logit_bias' in config:  # if logit scale is given, use it
            self.logit_bias = nn.Parameter(torch.tensor(config['logit_bias'], dtype=torch.float),
                                            requires_grad=config.get('train_logit_bias', False))
        else:
            self.logit_bias = nn.Parameter(torch.ones([]) * (-10))
        self.all_gather = config.get('all_gather', False)

    def forward_one_sided(self, left_emb, right_emb):
        if self.all_gather and self.world_size > 1:
            right_emb = GatherLayer.apply(right_emb, dist.group.WORLD, self.rank)
            assert right_emb.shape[0] == left_emb.shape[0] * self.world_size
            pos_idx = torch.arange(self.rank*left_emb.shape[0], (self.rank+1)*left_emb.shape[0], device=self.device)
        else:
            pos_idx = torch.arange(left_emb.shape[0], device=self.device)
        y = 2 * F.one_hot(pos_idx, num_classes=right_emb.shape[0]) - 1
        logits = self.logit_scale * (left_emb @ right_emb.T) + self.logit_bias
        loss = F.logsigmoid(logits * y)
        return -loss.sum(1).mean()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith('detach(') and name.endswith(')'):
                return embs[name[len('detach('):-len(')')]].detach()
            else:
                raise NotImplementedError
        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            left_emb = F.normalize(left_emb, dim=-1)
            right_emb = F.normalize(right_emb, dim=-1)
            loss += lossterm['factor']*self.forward_one_sided(left_emb, right_emb)
        return loss


class AlignmentLoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def forward_one_sided(self, left_emb, right_emb, right_ids=None):
        return 2 - 2 * (left_emb*right_emb).sum(-1).mean()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith('detach(') and name.endswith(')'):
                return embs[name[len('detach('):-len(')')]].detach()
            else:
                raise NotImplementedError
        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            left_emb = F.normalize(left_emb, dim=-1)
            right_emb = F.normalize(right_emb, dim=-1)
            loss += lossterm['factor']*self.forward_one_sided(left_emb, right_emb, lossterm.get('rightids'))
        return loss


class SimSiamLoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        modules = []
        dims = list(map(int, config['projection'].split(',')))
        for i in range(1, len(dims)):
            modules.append(nn.GELU())
            modules.append(nn.Linear(dims[i-1], dims[i]))
        self.proj = nn.Sequential(*modules)

    def forward_one_sided(self, left_proj, right_emb, right_ids=None):
        return 2 - 2 * (F.normalize(left_proj, dim=-1)*F.normalize(right_emb.detach(), dim=-1)).sum(-1).mean()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith('detach(') and name.endswith(')'):
                return embs[name[len('detach('):-len(')')]].detach()
            else:
                raise NotImplementedError
        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            left_proj, right_proj = self.proj(left_emb), self.proj(right_emb)
            lossterms = [self.forward_one_sided(left_proj, right_emb),
                         self.forward_one_sided(right_proj, left_emb)]
            loss += lossterm['factor']*torch.stack(lossterms).mean()
        return loss


class UniformityLoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def forward_one_sided(self, left_emb, right_emb, right_ids=None):
        sq_dists = (2 - 2 * (left_emb @ right_emb.T)).flatten()
        return sq_dists.mul(-2).exp().mean().log()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith('detach(') and name.endswith(')'):
                return embs[name[len('detach('):-len(')')]].detach()
            else:
                raise NotImplementedError
        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            left_emb = F.normalize(left_emb, dim=-1)
            right_emb = F.normalize(right_emb, dim=-1)
            loss += lossterm['factor']*self.forward_one_sided(left_emb, right_emb, lossterm.get('rightids'))
        return loss


class DictSelectLoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def forward(self, embs, lossterms):
        res = embs
        for k in lossterms:
            res = res[k]
        return res


class RetroMAELoss(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        self.config = config
        self.decoder = decoder

        self.mask_rate = config.get('mask_rate', 0.5)
        self.min_label_tokenid = config.get('min_label_tokenid', 999)
        self.mask_tokenid = config.get('mask_tokenid', 103)
        # whether to append embedding as 1 of the keys
        self.keys_add_enc = config.get('keys_add_enc', True)
        # whether to replace query embeddings instead of mask with embeddings
        self.query_ovrd_enc = config.get('query_ovrd_enc', True)
        # whether to add position information for query side in decoder
        self.dec_query_pos = config.get('dec_query_pos', True)
        # whether to add position information for hints/ keys for decoder
        self.dec_keys_pos = config.get('dec_keys_pos', True)

    def forward_direction(self, input, direction, cache={}):
        if f'{direction[0]}feat' not in cache:
            import pdb; pdb.set_trace()
            try:
                cache[f'{direction[0]}feat'] = self.encoder(input[f'{direction[0]}feat'])
            except:
                import pdb; pdb.set_trace()
        encoder_output = cache[f'{direction[0]}feat']

        decoder_feat = input[f'{direction[1]}feat']

        dec_am = decoder_feat['attention_mask'].unsqueeze(1)*decoder_feat['attention_mask'].unsqueeze(2)
        dec_am.diagonal(dim1=-1, dim2=-2).zero_()
        dec_am = dec_am*(torch.rand(dec_am.shape, device=dec_am.device)>self.mask_rate)

        label = decoder_feat['input_ids'].clone()
        label[label<self.min_label_tokenid]=-100

        if self.keys_add_enc:
            input_keys = self.decoder.get_token_emb(decoder_feat['input_ids'])
            input_keys = torch.cat([encoder_output.unsqueeze(1), input_keys[:, 1:]], dim=1)
        else:
            input_keys = self.decoder.get_token_emb(decoder_feat['input_ids'])

        if self.query_ovrd_enc:
            input_ids = encoder_output.unsqueeze(1).repeat(1, dec_am.shape[1], 1)
        else:
            input_ids = decoder_feat['input_ids'].clone()
            input_ids[:] = self.mask_tokenid

        decoder_input = {
            'input_keys': input_keys,
            'input_ids': input_ids,
            'attention_mask': dec_am
        }
        # make all positions 0, otherwise we use index as positions as all tokens are in order
        if not self.dec_query_pos:
            decoder_input['pos'] = torch.zeros_like(decoder_feat['input_ids'])
        if not self.dec_keys_pos:
            decoder_input['pos_keys'] = torch.zeros_like(decoder_feat['input_ids'])

        decoder_output = self.decoder(decoder_input)

        return decoder_output, label, cache

    def forward(self, embs, lossterms):
        # loss terms should have directions information
        
        res, cache = {}, {}
        for direction in self.directions:
            logitname, labelname = f'{direction}logits', f'{direction}label'
            res[logitname], res[labelname], cache = self.forward_direction(input, direction, cache)
        return res



class MAELoss(nn.Module):
    """
        Normal cross entropy loss
    """
    def __init__(self, encoders, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.decoder = create_encoder(config['decoder'], {})
        if config.get('tie_decoder_vocab', None) is not None:
            encoder_name = config['tie_decoder_vocab']
            logger.info('decoder vocab weights tied to {encoder_name}')
            self.decoder.vocab_mat().weight = encoders[encoder_name].vocab_mat().weight

        if config.get('tie_decoder_pos', None) is not None:
            encoder_name = config['tie_decoder_pos']
            logger.info('decoder pos embedding weights tied to {encoder_name}')
            self.decoder.pos_emb().weight = encoders[encoder_name].pos_emb().weight

        if config['mae_type'] == 'retromae':
            self.self = RetroMAELoss(self.decoder, config['mae_params'])
        else:
            raise NotImplementedError

        self.loss = nn.CrossEntropyLoss()

    def forward(self, embs, lossterms):
        self.self(embs, lossterms)


class CELoss(nn.Module):
    """
        Normal cross entropy loss
    """
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.loss = nn.CrossEntropyLoss()

    def forward(self, embs, lossterms):
        def get_emb(embs, name):
            res = embs
            for n in name.split('.'):
                res = res[n]
            return res
        def fetch_emb(name):
            if name.startswith('detach(') and name.endswith(')'):
                return get_emb(embs, name[len('detach('):-len(')')]).detach()
            else:
                return get_emb(embs, name)
        loss = 0
        for lossterm in lossterms:
            logits, label = fetch_emb(lossterm['lefttarget']), fetch_emb(lossterm['righttarget'])
            loss += lossterm['factor']*self.loss(logits.view((-1, logits.shape[-1])), label.ravel())
        return loss


class DictLoss(nn.Module):
    def __init__(self, config, encoders, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.losses = {}
        for key, _conf in config.items():
            self.losses[key] = create_loss(_conf, encoders, rank, world_size, device)
        self.losses = nn.ModuleDict(self.losses)

    def clip_logits(self):
        for l in self.losses.values():
            if hasattr(l, 'clip_logits'):
                l.clip_logits()

    def forward(self, embs, targets):
        if targets['type'] == 'sum':  # for training, returns scalar
            loss, lossterms_agg = 0, {}
            for lossterm in targets['terms']:
                if lossterm['lossname'] in lossterms_agg:
                    lossterms_agg[lossterm['lossname']].append(lossterm)
                else:
                    lossterms_agg[lossterm['lossname']] = [lossterm]
            for name, terms in lossterms_agg.items():
                loss += self.losses[name](embs, terms)
            return loss
        elif targets['type'] == 'dict':  # for loss evaluation, returns dict of losses
            def run_dict_forward(embs, targets, suffix=''):
                """
                For each element of targets, run the forward individually
                """
                return {key+suffix: self.forward(embs, targets[key]) for key in targets}
            targets = targets['contents']
            res = {}
            if 'max_batch_sizes' in targets:
                new_targets = {key: targets[key] for key in targets if key != 'max_batch_sizes'}
                for max_batch_size in targets['max_batch_sizes']:
                    suf = '_'+str(max_batch_size)
                    embs_splits = split_by_max_batch_size(embs, max_batch_size//self.world_size)
                    losses = [run_dict_forward(embs_split, new_targets, suffix=suf)
                              for embs_split in embs_splits]
                    res.update({key: torch.stack([loss[key] for loss in losses]).mean() for key in losses[0]})
            else:
                res.update(run_dict_forward(embs, targets))
            return res
        else:
            raise NotImplementedError
