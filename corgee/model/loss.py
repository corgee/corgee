import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from main.main_utils import split_by_max_batch_size
from model.model_helpers import GatherLayer

logger = logging.getLogger(__name__)


def create_loss(config, encoders, rank, world_size, device):
    assert isinstance(config, dict)
    if "type" not in config:
        return DictLoss(config, encoders, rank, world_size, device)
    elif config["type"] == "nce":
        return NCELoss(config, rank, world_size, device)
    else:
        raise NotImplementedError


class NCELoss(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        if "logit_scale" in config:  # if logit scale is given, use it
            self.logit_scale = nn.Parameter(
                torch.tensor(config["logit_scale"], dtype=torch.float),
                requires_grad=config.get("train_logit_scale", False),
            )
        else:  # else initialize it to log(1/0.07) and train it
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sim = "cosine"
        self.maxbatch_sim = config.get("maxbatch_sim", None)
        self.logit_scale_max = config.get("logit_scale_max", None)
        self.all_gather = config.get("all_gather", True)
        self.both_sides = config.get("both_sides", False)
        self.ignore_score_eq_pos = config.get("ignore_score_eq_pos", False)
        self.ignore_score_gteq_pos = config.get("ignore_score_gteq_pos", False)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.neg = config.get("neg", None)
        self.matryoshka = config.get("matryoshka", None)  # dict of dim to weight
        if self.neg:
            assert self.neg["type"] in ["random", "topk"]
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def clip_logits(self):
        if self.logit_scale_max is not None:
            self.logit_scale.data.clip_(None, self.logit_scale_max)

    def compute_sim(self, left_emb, right_emb):
        return F.normalize(left_emb, dim=-1) @ F.normalize(right_emb, dim=-1).T

    def compute_loss(self, left_emb, right_emb, pos_idx, right_ids):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * self.compute_sim(left_emb, right_emb)
        if self.neg:
            pos_idx_2d = pos_idx.unsqueeze(1)
            # pos logits [bsz, 1]
            pos_logits = torch.gather(logits, 1, pos_idx_2d)
            # [bsz, #cand] with positives at -100
            neg_logits_all = torch.scatter(
                logits,
                1,
                pos_idx_2d,
                torch.zeros_like(pos_idx_2d).to(logits.dtype) - 100,
            )
            # [bsz, #cand-1] all negatives sorted by logit
            negidx_sorted = torch.argsort(neg_logits_all, descending=True)[:, :-1]
            if self.neg["type"] == "topk":
                neg_idx = negidx_sorted[:, : self.neg["num"]]
            elif self.neg["type"] == "random":
                rand_indices = torch.topk(
                    torch.randn_like(negidx_sorted.to(torch.float)), self.neg["num"]
                ).indices
                neg_idx = torch.gather(negidx_sorted, 1, rand_indices)
            neg_logits = torch.gather(logits, 1, neg_idx)
            # [batch_size, num_negatives+1]
            logits = torch.cat((pos_logits, neg_logits), axis=1)
            pos_idx = torch.zeros_like(pos_idx)
        if right_ids is not None:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = right_ids[pos_idx_2d] == right_ids.unsqueeze(0)
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        if self.ignore_score_eq_pos:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = logits == torch.gather(logits, 1, pos_idx_2d)
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        if self.ignore_score_gteq_pos:
            pos_idx_2d = pos_idx.unsqueeze(1)
            mask = logits >= torch.gather(logits, 1, pos_idx_2d)
            mask.scatter_(1, pos_idx_2d, False)
            logits[mask] = -1e4
        return self.criterion(logits, pos_idx)

    def forward_one_sided(self, left_emb, right_emb, right_ids=None):
        if self.all_gather and self.world_size > 1:
            right_emb = GatherLayer.apply(right_emb, dist.group.WORLD, self.rank)
            if right_ids is not None:
                right_ids = GatherLayer.apply(right_ids, dist.group.WORLD, self.rank)
            assert right_emb.shape[0] == left_emb.shape[0] * self.world_size
            pos_idx = torch.arange(
                self.rank * left_emb.shape[0],
                (self.rank + 1) * left_emb.shape[0],
                device=self.device,
            )
        else:
            pos_idx = torch.arange(left_emb.shape[0], device=self.device)
        if len(right_emb.shape) == 3:
            # flatten such that positives come first and negatives after them
            right_emb = right_emb.transpose(0, 1).reshape(-1, right_emb.shape[-1])
        if self.matryoshka is None:
            return self.compute_loss(left_emb, right_emb, pos_idx, right_ids)
        else:
            lossterms_eval = []
            # 64:1, 128: 1, 256:1
            for dim, weight in self.matryoshka.items():
                lossterms_eval.append(
                    weight
                    * self.compute_loss(
                        left_emb[..., :dim], right_emb[..., :dim], pos_idx, right_ids
                    )
                )
            return torch.stack(lossterms_eval).sum()

    def forward(self, embs, lossterms):
        def fetch_emb(name):
            if name in embs:
                return embs[name]
            elif name.startswith("detach(") and name.endswith(")"):
                return embs[name[len("detach(") : -len(")")]].detach()
            else:
                raise NotImplementedError

        loss = 0
        for lossterm in lossterms:
            left_emb, right_emb = (
                fetch_emb(lossterm["lefttarget"]),
                fetch_emb(lossterm["righttarget"]),
            )
            lossterms_eval = [
                self.forward_one_sided(left_emb, right_emb, lossterm.get("rightids"))
            ]
            if self.both_sides:
                lossterms_eval.append(self.forward_one_sided(right_emb, left_emb))
            loss += lossterm["factor"] * torch.stack(lossterms_eval).mean()
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
        for loss in self.losses.values():
            if hasattr(loss, "clip_logits"):
                loss.clip_logits()

    def forward(self, embs, targets):
        if targets["type"] == "sum":  # for training, returns scalar
            loss, lossterms_agg = 0, {}
            for lossterm in targets["terms"]:
                if lossterm["lossname"] in lossterms_agg:
                    lossterms_agg[lossterm["lossname"]].append(lossterm)
                else:
                    lossterms_agg[lossterm["lossname"]] = [lossterm]
            for name, terms in lossterms_agg.items():
                loss += self.losses[name](embs, terms)
            return loss
        elif targets["type"] == "dict":  # for loss evaluation, returns dict of losses

            def run_dict_forward(embs, targets, suffix=""):
                """
                For each element of targets, run the forward individually
                """
                return {
                    key + suffix: self.forward(embs, targets[key]) for key in targets
                }

            targets = targets["contents"]
            res = {}
            if "max_batch_sizes" in targets:
                new_targets = {
                    key: targets[key] for key in targets if key != "max_batch_sizes"
                }
                for max_batch_size in targets["max_batch_sizes"]:
                    suf = "_" + str(max_batch_size)
                    embs_splits = split_by_max_batch_size(
                        embs, max_batch_size // self.world_size
                    )
                    losses = [
                        run_dict_forward(embs_split, new_targets, suffix=suf)
                        for embs_split in embs_splits
                    ]
                    res.update(
                        {
                            key: torch.stack([loss[key] for loss in losses]).mean()
                            for key in losses[0]
                        }
                    )
            else:
                res.update(run_dict_forward(embs, targets))
            return res
        else:
            raise NotImplementedError
