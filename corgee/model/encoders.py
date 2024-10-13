import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.me5_adapters import fuse_glu_into_me5, fuse_rope_into_me5
from model.model_helpers import init_weights
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def create_encoder(enc_conf, encoders):
    if not isinstance(enc_conf, dict) or "type" not in enc_conf:
        raise NotImplementedError
    enc_type = enc_conf["type"]
    if enc_type == "combiner":
        encoder = Combiner(enc_conf, encoders)
    elif enc_type == "sequential":
        encoder = CustomSequential(*[encoders[k] for k in enc_conf["list"]])
    else:
        encoder = SingleEncoder(enc_conf)

    # This support would be removed in the future
    # TODO: Remove this support, would not be backward compatible
    if enc_conf.get("projection", None) is not None:
        encoder = CustomSequential(
            encoder, LinearProjection(enc_conf["projection"], in_dim=encoder.out_dim)
        )

    load = enc_conf.get("load", None)
    if load is not None:
        sd = torch.load(load["path"], map_location="cpu")
        # import code; code.interact(local=locals())
        if not load.get("state_dict_at_root", False):
            sd = sd["model_state_dict"]
        if "transformations" in load:
            for trf in load["transformations"]:
                assert isinstance(trf, dict) and len(trf) == 1
                if "replace_start" in trf:
                    find, replace = trf["replace_start"].split("=>")
                    sd = {
                        (replace + k[len(find) :] if k.startswith(find) else k): v
                        for k, v in sd.items()
                    }
                elif "filter" in trf:
                    sd_filter = {}
                    for pattern in trf["filter"].split("|"):
                        sd_filter.update(
                            {k: v for k, v in sd.items() if k.startswith(pattern)}
                        )
                    sd = sd_filter
                elif "remove_start" in trf:
                    sd = {
                        k: v
                        for k, v in sd.items()
                        if not k.startswith(trf["remove_start"])
                    }
                elif "replace" in trf:
                    find, replace = trf["replace"].split("=>")
                    sd = {k.replace(find, replace): v for k, v in sd.items()}
                else:
                    raise NotImplementedError
        res = encoder.load_state_dict(sd, strict=load.get("strict", True))
        if not load.get("strict"):
            logger.info(f"missing keys {res.missing_keys}")
            logger.info(f"unexpected keys {res.unexpected_keys}")

    return encoder


class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self._custom_modules = []
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            self._custom_modules.append(module)

    def __getitem__(self, i):
        return self._custom_modules[i]

    def __iter__(self):
        return iter(self._modules.values())

    def get_token_emb(self, *args, **kwargs):
        return self._custom_modules[0].get_token_emb(*args, **kwargs)

    def pos_emb(self, *args, **kwargs):
        return self._custom_modules[0].pos_emb(*args, **kwargs)

    def vocab_mat(self, *args, **kwargs):
        return self._custom_modules[0].vocab_mat(*args, **kwargs)

    def forward(self, input, return_layerwise=False):
        # TODO: Currently a hacky solution to not apply projection on layerwise output
        for idx, module in enumerate(self):
            if return_layerwise and idx == 0:
                input, layerwise = module(input, return_layerwise=True)
            else:
                input = module(input)
        if return_layerwise:
            return input, layerwise
        else:
            return input


class STransformerInputLayer(nn.Module):
    def __init__(self, transformer="roberta-base"):
        super(STransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.out_dim = self.transformer.get_sentence_embedding_dimension()

    def encode(self, data):
        if data["input_ids"].ndim > 2:
            shape = data["input_ids"].shape
            embs = self.transformer(
                {
                    "input_ids": data["input_ids"].view((-1, shape[-1])),
                    "attention_mask": data["attention_mask"].view((-1, shape[-1])),
                }
            )["sentence_embedding"]
            return embs.view((*shape[:-1], embs.shape[-1]))
        else:
            return self.transformer(data)["sentence_embedding"]

    def forward(self, data):
        if "mask" in data:
            mask = data["mask"]
            emb = torch.zeros(
                (data["input_ids"].shape[0], self.out_dim),
                dtype=torch.float32,
                device=mask.device,
            )
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class weightpooler(nn.Module):
    def __init__(self, hidden_size):
        super(weightpooler, self).__init__()
        self.weighting = nn.Linear(hidden_size, 1)

    def forward(self, term_tensor, mask):
        weights = self.weighting(term_tensor)
        weights = weights + ((mask - 1).type(torch.float) / 1e-8).unsqueeze(2).type(
            weights.dtype
        )
        weights = F.softmax(weights, dim=1)
        return (term_tensor * weights).sum(dim=1)


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        if isinstance(config, str):
            self.cls_id = None
            self.pooling_mode = config
        elif isinstance(config, dict):
            self.cls_id = config.get("cls_id", None)
            self.pooling_mode = config["mode"]
            if self.pooling_mode == "weighted":
                self.weight = nn.Linear(config["dim"], 1)
            elif self.pooling_mode == "multihead_weighted":
                self.dim = config["dim"]
                self.num_heads = config["num_heads"]
                self.per_head_dim = self.dim // self.num_heads
                self.weight = nn.Linear(self.dim, self.num_heads)
        else:
            raise NotImplementedError
        assert self.pooling_mode in [
            "mean",
            "max",
            "min",
            "cls",
            "weighted",
            "multihead_weighted",
            "all",
        ]

    def forward(self, hidden_states, attention_mask, input_ids=None):
        if self.pooling_mode == "mean":
            return torch.sum(
                hidden_states * attention_mask.unsqueeze(-1), 1
            ) / torch.clamp(attention_mask.sum(1).unsqueeze(-1), min=1e-9)
        elif self.pooling_mode == "max":
            return torch.max(hidden_states, 1)[0]
        elif self.pooling_mode == "min":
            return torch.min(hidden_states, 1)[0]
        elif self.pooling_mode == "cls":
            if self.cls_id is None:
                return hidden_states[:, 0]
            else:
                return hidden_states[torch.where(input_ids == self.cls_id)]
        elif self.pooling_mode == "weighted":
            weights = self.weight(hidden_states)
            weights = weights + (
                (attention_mask - 1).type(torch.float) / 1e-8
            ).unsqueeze(-1).type(weights.dtype)
            weights = F.softmax(weights, dim=1)
            return (hidden_states * weights).sum(dim=1)
        elif self.pooling_mode == "multihead_weighted":
            weights = self.weight(hidden_states)  # [B, T, num_heads]
            hidden_states = hidden_states.view(
                (*hidden_states.shape[:-1], self.num_heads, self.per_head_dim)
            )  # [B, T, num_heads, per_head_dim]
            weights += (
                ((attention_mask - 1).type(torch.float) / 1e-8)
                .unsqueeze(-1)
                .type(weights.dtype)
            )
            weights = F.softmax(weights, dim=1)  # [B, T, num_heads] normalize along T
            return (
                (hidden_states * weights.unsqueeze(-1)).sum(dim=1).view((-1, self.dim))
            )
        elif self.pooling_mode == "all":
            # return attention_mask.unsqueeze(2)*hidden_states
            # TODO: check if we need this attention mask multiplication
            return hidden_states
        else:
            raise NotImplementedError


class Residual(nn.Module):
    def __init__(self, dim, _type):
        super(Residual, self).__init__()
        self.dim = dim
        self.type = _type
        self.mat = nn.Linear(dim, dim)

    def forward(self, inp):
        if self.type == "relu":
            return inp + F.relu(self.mat(inp))
        elif self.type == "gelu":
            return inp + F.gelu(self.mat(inp))
        elif self.type == "pre_relu":
            return inp + self.mat(F.relu(inp))
        elif self.type == "pre_gelu":
            return inp + self.mat(F.gelu(inp))
        elif self.type == "relu_norm":
            return F.normalize(inp + F.relu(self.mat(inp)), dim=-1)
        elif self.type == "gelu_norm":
            return F.normalize(inp + F.gelu(self.mat(inp)), dim=-1)
        elif self.type == "relu_resnorm":
            return inp + F.normalize(F.relu(self.mat(inp)), dim=-1)
        elif self.type == "gelu_resnorm":
            return inp + F.normalize(F.gelu(self.mat(inp)), dim=-1)
        else:
            raise NotImplementedError


class LinearProjection(nn.Module):
    def __init__(self, config, in_dim=None):
        super(LinearProjection, self).__init__()

        self.config = config
        self.in_dim, self.out_dim, self.num_layers = in_dim, None, 0
        self.residual_type, self.init, self.normalize_input = "relu", None, False
        self.add_first_dims = False
        self.first_dims_only = False

        if isinstance(config, int):
            self.out_dim = config
        elif isinstance(config, dict):
            self.in_dim = config.get("in_dim", self.in_dim)
            self.out_dim = config["out_dim"]
            self.num_layers = config.get("num_layers", 0)
            self.residual_type = config.get("residual_type", "relu")
            self.init = config.get("init", None)
            self.normalize_input = config.get("normalize_input", False)
            self.add_first_dims = config.get("add_first_dims", False)
            self.first_dims_only = config.get("first_dims_only", False)
            if self.first_dims_only:
                assert not self.add_first_dims
                assert self.num_layers == 0
        else:
            raise NotImplementedError

        assert self.in_dim is not None and self.out_dim is not None

        modules = [
            Residual(self.in_dim, self.residual_type) for _ in range(self.num_layers)
        ]
        if not self.first_dims_only:
            modules.append(nn.Linear(self.in_dim, self.out_dim))
        self.linear_layers = nn.Sequential(*modules)

        if self.init is not None:

            def randomize_f(module_):
                return init_weights(module_, self.init["range"], self.init["clamp"])

            self.linear_layers.apply(randomize_f)

    def project(self, embs):
        if self.first_dims_only:
            return embs[..., : self.out_dim]
        proj_embs = self.linear_layers(embs)
        if self.add_first_dims:
            proj_embs += embs[..., : self.out_dim]
        return proj_embs

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            embs, mask = data, None
        elif isinstance(data, dict):
            embs, mask = data["embs"], data.get("mask", None)
        else:
            raise NotImplementedError

        if self.normalize_input:
            embs = F.normalize(embs, dim=-1)
        if mask is not None:
            embs = self.project(embs)
            embs[~mask] = 0
            return embs
        else:
            return self.project(embs)


class SingleEncoder(nn.Module):
    def __init__(self, enc_conf):
        super().__init__()
        self.enc_conf = enc_conf

        enc_type = enc_conf["type"]

        if enc_type == "sbert":
            enc_name = enc_conf["desc"]
            encoder = STransformerInputLayer(SentenceTransformer(enc_name))
            if enc_conf.get("trim_layers", None):
                if enc_conf.get("roberta", False):
                    trimmed_layers = encoder.transformer[0].auto_model.encoder.layer[
                        : -enc_conf["trim_layers"]
                    ]
                    encoder.transformer[0].auto_model.encoder.layer = trimmed_layers
                else:
                    trimmed_layers = encoder.transformer[
                        0
                    ].auto_model.transformer.layer[: -enc_conf["trim_layers"]]
                    encoder.transformer[0].auto_model.transformer.layer = trimmed_layers
            if enc_conf.get("fuse_rope_into_me5"):
                fuse_rope_into_me5(encoder.transformer)
            if enc_conf.get("fuse_glu_into_me5"):  # fuse before randomize intentionally
                fuse_glu_into_me5(encoder.transformer)
            if enc_conf.get("randomize"):
                encoder.apply(partial(init_weights, std=enc_conf["randomize"]))
            if enc_conf.get("change_mean_to_cls_pooler"):
                encoder.transformer[1].pooling_mode_mean_tokens = False
                encoder.transformer[1].pooling_mode_cls_token = True
            logger.info(encoder.transformer[1].get_pooling_mode_str())
            out_dim = encoder.transformer.get_sentence_embedding_dimension()
        else:
            raise NotImplementedError

        self.encoder = encoder
        self.out_dim = out_dim

    # get_token_emb, pos_emb, vocab_mat might fail for some encoders
    # TODO: handle this gracefully
    def get_token_emb(self, *args, **kwargs):
        return self.encoder.get_token_emb(*args, **kwargs)

    def pos_emb(self, *args, **kwargs):
        return self.encoder.pos_emb(*args, **kwargs)

    def vocab_mat(self, *args, **kwargs):
        return self.encoder.vocab_mat(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class Combiner(nn.Module):
    def __init__(self, config, input_encoders):
        super().__init__()
        self.config = config
        assert config["type"] == "combiner"

        encoders_list = config["encoders"]
        if isinstance(encoders_list, str):
            encoders_list = [x.strip() for x in encoders_list.split(",")]
        elif not isinstance(encoders_list, list):
            raise NotImplementedError
        self.encoders = nn.ModuleList(input_encoders[k] for k in encoders_list)

        if config["combiner_type"] == "mean":
            self.type = "mean"
        elif config["combiner_type"] == "weighted_mean":
            self.type = "weighted_mean"
            self.weights = torch.tensor(list(map(float, config["weights"].split(","))))[
                :, None, None
            ]
        elif config["combiner_type"] == "norm_mean":
            self.type = "norm_mean"
        elif config["combiner_type"] == "self_attention":
            self.type = "self_attention"
            self.attn = nn.MultiheadAttention(config["embed_dim"], config["num_heads"])
        else:
            raise NotImplementedError

    def forward(self, listfeat):
        inp = [
            encoder(mode_feat) for encoder, mode_feat in zip(self.encoders, listfeat)
        ]
        if (
            self.type == "mean"
            or self.type == "norm_mean"
            or self.type == "weighted_mean"
        ):
            inp = torch.stack(inp)
            if self.type == "norm_mean":
                inp = F.normalize(inp, dim=-1)
            elif self.type == "weighted_mean":
                if self.weights.device != inp.device:
                    self.weights = self.weights.to(inp.device)
                inp = inp * self.weights
            if any(["mask" in feat_mode for feat_mode in listfeat]):
                masks = []
                for feat_mode in listfeat:
                    masks.append(
                        feat_mode.get(
                            "mask",
                            torch.ones(
                                len(inp[0]), dtype=torch.bool, device=inp.device
                            ),
                        )
                    )
                masks = torch.stack(masks)
                return (inp * masks.unsqueeze(2)).sum(0) / masks.sum(0).unsqueeze(1)
            else:
                return inp.mean(0)
        elif self.type == "self_attention":
            inp = torch.stack(inp)
            masks = []
            for feat_mode in listfeat:
                masks.append(
                    feat_mode.get(
                        "mask",
                        torch.ones(len(inp[0]), dtype=torch.bool, device=inp.device),
                    )
                )
            masks = torch.stack(masks)
            attn_emb = self.attn(
                inp, inp, inp, key_padding_mask=~masks.T, need_weights=False
            )[0]
            return (attn_emb * masks.unsqueeze(2)).sum(0) / masks.sum(0).unsqueeze(1)
        else:
            raise NotImplementedError
