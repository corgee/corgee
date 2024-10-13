import math
import types
import torch
import logging
import torch.nn as nn
from functools import partial


logger = logging.getLogger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, posq, posk):
    cosq = cos[posq].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sinq = sin[posq].unsqueeze(1)
    q_embed = (q * cosq) + (rotate_half(q) * sinq)
    cosk = cos[posk].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sink = sin[posk].unsqueeze(1)
    k_embed = (k * cosk) + (rotate_half(k) * sink)
    return q_embed, k_embed


# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, device, dtype, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=dtype),
            self.sin_cached[:seq_len].to(dtype=dtype),
        )



def xlmroberta_forward_self_attention_with_rope(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
        rotary_emb = None
    ):

    if encoder_hidden_states is not None or past_key_value is not None or self.is_decoder:
        raise NotImplementedError

    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))

    cos, sin = rotary_emb(key_layer.device, key_layer.dtype, seq_len=key_layer.shape[2])
    pos_ids = torch.arange(key_layer.shape[2]).unsqueeze(0)
    query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, pos_ids, pos_ids)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


def fuse_rope_into_me5(st):
    st[0].auto_model.embeddings.position_embedding_type = 'rope'
    head_dim = st[0].auto_model.config.hidden_size//st[0].auto_model.config.num_attention_heads
    max_position_embeddings = st[0].auto_model.config.max_position_embeddings
    rotary_emb = LlamaRotaryEmbedding(
        head_dim,
        max_position_embeddings=max_position_embeddings,
        base=10000.0,
        device='cuda'
    )
    for layer in st[0].auto_model.encoder.layer:
        attn_layer = layer.attention.self
        forward_fn = partial(xlmroberta_forward_self_attention_with_rope, rotary_emb=rotary_emb)
        attn_layer.forward = types.MethodType(forward_fn, attn_layer)


class XLMRobertaIntermediateGLU(nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.dense = nn.Linear(orig_layer.dense.in_features, orig_layer.dense.out_features, 
                               bias=True)
        self.dense.weight.data = self.orig_layer.dense.weight.data.clone()
        self.dense.bias.data = self.orig_layer.dense.bias.data.clone()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.orig_layer(hidden_states) * self.dense(hidden_states)


def fuse_glu_into_me5(st):
    logger.info('fusing glu')
    for layer in st[0].auto_model.encoder.layer:
        layer.intermediate = XLMRobertaIntermediateGLU(layer.intermediate)