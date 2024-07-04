import math
import types
import torch
import logging
import torch.nn as nn
from functools import partial
from model.transformer.arch import apply_rotary_pos_emb, LlamaRotaryEmbedding


logger = logging.getLogger(__name__)


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
