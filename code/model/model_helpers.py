
import torch
import torch.nn as nn
import torch.distributed as dist


def init_weights_data(weight, std, clamp=None):
    if clamp == 0 or clamp is None:
        torch.nn.init.normal_(weight, mean=0.0, std=std)
    else:
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-clamp*std, b=clamp*std)


def init_weights(module, std, clamp=None, std_emb=None):
    if std_emb is None:
        std_emb = std
    if isinstance(module, nn.Linear):
        # clamp support for https://github.com/pytorch/pytorch/pull/5617
        init_weights_data(module.weight, std, clamp)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        init_weights_data(module.weight, std_emb, clamp)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def count_parameters(model):
    total_params = 0
    embedding_params = 0
    non_embedding_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if isinstance(param, nn.Embedding) or 'embedding' in name:
            embedding_params += num_params
        else:
            non_embedding_params += num_params
    return total_params, embedding_params, non_embedding_params


class GatherLayer(torch.autograd.Function):
    """
        :class:`GatherLayer` is a module wrapper that realizes backward op in all_gather
        Usage:
        feat_global = torch.cat(all_gather(feat, group), 0)
        # equals to
        feat_global = GatherLayer.apply(feat, group, rank)
    """

    @staticmethod
    def forward(ctx, tensor, group, rank):
        ctx.batch_size = tensor.shape[0]
        ctx.group = group
        ctx.rank = rank

        gathered_tensor = [torch.zeros_like(tensor) for _ in
                           range(dist.get_world_size(group))]

        dist.all_gather(gathered_tensor, tensor.contiguous(), group=group)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False,
                        group=ctx.group)

        idx_from = ctx.rank * ctx.batch_size
        idx_to = (ctx.rank + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to], None, None


class GatherLayerStack(torch.autograd.Function):
    """
        :class:`GatherLayer` is a module wrapper that realizes backward op in all_gather
        Usage:
        feat_global = torch.stack(all_gather(feat, group), 0)
        # equals to
        feat_global = GatherLayer.apply(feat, group, rank)
    """

    @staticmethod
    def forward(ctx, tensor, group, rank):
        ctx.group = group
        ctx.rank = rank

        gathered_tensor = [torch.zeros_like(tensor) for _ in
                           range(dist.get_world_size(group))]

        dist.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False,
                        group=ctx.group)

        return grad_input[ctx.rank], None, None
