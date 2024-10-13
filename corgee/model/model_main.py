import logging
from contextlib import ExitStack, contextmanager

import torch
import torch.nn as nn
from model.encoders import create_encoder
from model.loss import create_loss
from model.model_helpers import count_parameters

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, config, rank, world_size, device):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.encoders = {}
        self.clone_encoders = {}
        for key, enc_conf in config["encoders"].items():
            if isinstance(enc_conf, str):
                self.clone_encoders[key] = enc_conf
            else:
                self.encoders[key] = create_encoder(enc_conf, self.encoders)
                tot_p, emb_p, non_emb_p = count_parameters(self.encoders[key])
                logger.info(
                    f"encoder {key} #params emb {emb_p:.2e} non-emb {non_emb_p:.2e} total {tot_p:.2e}"
                )

        self.encoders = nn.ModuleDict(self.encoders)
        self.loss = None
        if "loss" in config:
            self.loss = create_loss(
                config["loss"], self.encoders, rank, world_size, device
            )
        self.find_unused_parameters = config.get("find_unused_parameters", False)
        self.bf16 = config.get("bf16", False)
        self.compile_encoders = config.get("compile_encoders", False)

        if self.bf16:
            self.to(dtype=torch.bfloat16)
        self.to(device)

        if world_size > 1:
            for k in self.encoders:
                if (
                    len(
                        list(
                            p for p in self.encoders[k].parameters() if p.requires_grad
                        )
                    )
                    > 0
                ):
                    self.encoders[k] = nn.parallel.DistributedDataParallel(
                        self.encoders[k],
                        device_ids=[device],
                        broadcast_buffers=False,
                        find_unused_parameters=self.find_unused_parameters,
                    )
                else:
                    logger.info(f"Skip DDP for {k} because it has no parameters")
            if isinstance(self.loss, nn.Module):
                if len(list(p for p in self.loss.parameters() if p.requires_grad)) > 0:
                    self.loss = nn.parallel.DistributedDataParallel(
                        self.loss,
                        device_ids=[device],
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                    )

        if self.compile_encoders:
            for k in self.encoders:
                self.encoders[k] = torch.compile(self.encoders[k])

    def encoder_forward(self, k, inputs, **kwargs):
        if k in self.encoders:
            encoder = self.encoders[k]
        else:
            encoder = self.encoders[self.clone_encoders[k]]
        return encoder(inputs, **kwargs)

    def forward(self, batch_data, return_layerwise=False):
        if return_layerwise:
            res = {}
            for k, v in batch_data.items():
                if v is not None:
                    res[k], layerwise = self.encoder_forward(
                        k, v, return_layerwise=True
                    )
                    # store the layerwise embeddings as key%%index in the dict
                    res.update(
                        {k + f"%%{i}": layer for i, layer in enumerate(layerwise)}
                    )
                else:
                    res[k] = None
            return res
        else:
            return {
                k: self.encoder_forward(k, v) if v is not None else None
                for k, v in batch_data.items()
            }

    @contextmanager
    def no_sync(self):
        if self.world_size > 1:
            with ExitStack() as stack:
                for k in self.encoders:
                    if isinstance(
                        self.encoders[k], nn.parallel.DistributedDataParallel
                    ):
                        stack.enter_context(self.encoders[k].no_sync())
                if isinstance(self.loss, nn.parallel.DistributedDataParallel):
                    stack.enter_context(self.loss.no_sync())
                yield
        else:
            yield
