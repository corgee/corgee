# Major chunks copied from https://github.com/karpathy/nanoGPT/blob/master/model.py
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from model.model_helpers import init_weights, init_weights_data


logger = logging.getLogger(__name__)


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


class Attention(nn.Module):
    def __init__(self, config, block_num):
        super().__init__()
        self.n_head = config.n_head
        self.n_head_embd = config.n_head_embd
        self.n_embd = config.n_embd

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = nn.Linear(self.n_embd, self.n_head_embd*self.n_head, bias=config.bias)
        self.c_attn_k = nn.Linear(self.n_embd, self.n_head_embd*self.n_head, bias=config.bias)
        self.c_attn_v = nn.Linear(self.n_embd, self.n_head_embd*self.n_head, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(self.n_head_embd*self.n_head, self.n_embd, bias=config.bias)
        # regularization
        self.dropout = config.dropout
        if len(config.attn_dropouts) > 0:
            self.dropout = list(map(float, config.attn_dropouts.split(',')))[block_num]
        self.attn_dropout = nn.Dropout(self.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = config.flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')\
            and self.dropout == 0.0
        self.softmax = nn.Softmax(-1)

        # placing rope emb here needs least code interaction with other layers but redundancy
        # TODO: explore a clean way to push it to transformer level
        self.rope_pos_emb = config.rope_pos_emb
        if config.rope_pos_emb:
            self.rotary_emb = LlamaRotaryEmbedding(
                config.n_head_embd,
                max_position_embeddings=config.block_size,
                base=config.rope_theta
            )

    def forward(self, x, xall, posx, posxall, mask):
        Bq, Tq, Cq = x.size()  # batch size query, sequence length, embedding dimensionality (n_embd)
        B, T, C = xall.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        assert Bq == B and Cq == C
        # mask: bool is B, T
        if len(mask.shape) == 2:
            mask = mask.view((B, 1, 1, T))
        elif len(mask.shape) == 3:
            mask = mask.view((B, 1, Tq, T))
        else:
            raise NotImplementedError

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn_q(x), self.c_attn_k(xall), self.c_attn_v(xall)

        q = q.view(B, Tq, self.n_head, self.n_head_embd).transpose(1, 2)  # (B, nh, Tq, hs)
        k = k.view(B, T, self.n_head, self.n_head_embd).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_head_embd).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_pos_emb:
            cos, sin = self.rotary_emb(v.device, v.dtype, seq_len=1+max(posx.max(), posxall.max()).to(torch.long))
            q, k = apply_rotary_pos_emb(q, k, cos, sin, posx, posxall)

        # causal self-attention; Self-attend: (B, nh, Tq, hs) x (B, nh, hs, T) -> (B, nh, Tq, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            raise NotImplementedError
            # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            #                                                      dropout_p=self.dropout, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, nh, Tq, T]
            att.masked_fill_(mask.expand_as(att), torch.tensor(torch.finfo(att.dtype).min))
            att = self.softmax(att)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, Tq, T) x (B, nh, T, hs) -> (B, nh, Tq, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, Tq, self.n_head*self.n_head_embd)

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, block_num):
        super().__init__()

        self.glu = config.glu
        if config.glu:
            self.c_fc = nn.Linear(config.n_embd, 2*config.mlp_hid_dim, bias=config.bias)
        else:
            self.c_fc = nn.Linear(config.n_embd, config.mlp_hid_dim, bias=config.bias)

        self.c_proj = nn.Linear(config.mlp_hid_dim, config.n_embd, bias=config.bias)
        
        if config.nonlin == 'gelu':
            self.non_lin = nn.GELU()
        elif config.nonlin == 'relu':
            self.non_lin = nn.ReLU()
        elif config.nonlin == 'silu':
            self.non_lin = nn.SiLU()
        else:
            raise NotImplementedError

        dropout = config.dropout
        if len(config.hid_dropouts) > 0:
            dropout = list(map(float, config.hid_dropouts.split(',')))[block_num]
        self.dropout = nn.Dropout(dropout)

        actvn_dropout = 0.0  # by default 0 in most setups, fairseq uses it so trying it out
        if len(config.actvn_dropouts) > 0:
            actvn_dropout = list(map(float, config.actvn_dropouts.split(',')))[block_num]
        self.actvn_dropout = nn.Dropout(actvn_dropout)

    def forward(self, x):
        if self.glu:
            nonlinx, linx = torch.tensor_split(self.c_fc(x), 2, dim=-1)
            x = self.non_lin(nonlinx)*linx
        else:
            x = self.non_lin(self.c_fc(x))
        x = self.actvn_dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, block_num):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-12)
        self.attn = Attention(config, block_num)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-12)
        self.mlp = MLP(config, block_num)
        self.prelayernorm = config.prelayernorm

    def forward(self, x, xk, pos, posk, mask):
        if self.prelayernorm:
            x = x + self.attn(self.ln_1(x), self.ln_1(xk), pos, posk, mask)
            x = x + self.mlp(self.ln_2(x))
            return x
        else:
            xatt = self.ln_1(x + self.attn(x, xk, pos, posk, mask))
            xmlp = self.ln_2(xatt + self.mlp(xatt))
            return xmlp, [xatt, xmlp]


@dataclass
class Config:

    # transformer level params
    block_size: int = 128  # needed for position embeddings, encoder decoder both
    vocab_size: int = 50304  # needed for both encoder and decoder
    pad_token_id: int = 0
    n_layer: int = 6
    n_embd: int = 768
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    init_std: float = 0.02
    init_clamp: int = 0
    prelayernorm: bool = False
    add_pos_emb: bool = True
    rope_pos_emb: bool = False
    rope_theta: float = 10000.0
    decoder: bool = False
    project_input: int = 0

    # sa block params
    n_head: int = 12
    n_head_embd: int = 64

    # mlp params
    mlp_hid_dim: int = 3072
    glu: bool = False
    nonlin: str = "gelu"
    
    # training time params
    dropout: float = 0.1  # one stop setting to set tokenization, attention, hidden dropouts
    tok_dropout: float = -1  # if >= 0 overrides tokenization dropout
    attn_dropouts: str = ""  # if not empty, num_layers comma separated floats. override layerwise attention dropout
    actvn_dropouts: str = ""  # if not empty, num_layers comma separated floats. override layerwise activation dropout
    hid_dropouts: str = ""  # if not empty, num_layers comma separated floats. override layerwise hidden dropout

    # implementation params
    flash: bool = False  # use flash attention implementation
    cls_optimize_last_layer: bool = False  # in last layer only run 0th token TODO


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = Config(**config['params'])
        self.config = config
        self.out_dim = self.config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id),
            ln=nn.LayerNorm(config.n_embd, eps=1e-12),
            drop=nn.Dropout(config.tok_dropout if config.tok_dropout >= 0.0 else config.dropout),
            h=nn.ModuleList([Block(config, block_num) for block_num in range(config.n_layer)]),
        ))

        if config.project_input > 0:
            self.transformer.update(dict(
                proj_in=nn.Linear(config.project_input, config.n_embd, bias=config.bias),
            ))

        if config.decoder:
            self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
            self.decoder.weight = self.transformer.wte.weight

        if config.add_pos_emb:
            self.transformer.update(dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),
            ))

        def _init_weights(module):
            init_weights(module, config.init_std, config.init_clamp)
        self.apply(_init_weights)

        # report number of parameters
        logger.info("number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters())/1e6,))

    def pos_emb(self):
        return self.transformer.wpe

    def vocab_mat(self):
        return self.transformer.wte

    def get_token_emb(self, tokens):
        return self.transformer.wte(tokens)

    def preprocess_input(self, input, pos):
        T = input.shape[1]
        assert T <= self.config.block_size, f"sequence length {T} > block size {self.config.block_size}"
        if pos is None:
            pos = torch.arange(0, T, dtype=torch.long, device=input.device).unsqueeze(0)  # shape (1, T)
        # convert to embedding dimension if needed
        if str(input.dtype).startswith('torch.int') or str(input.dtype).startswith('torch.uint'):
            input = self.transformer.wte(input)
        if self.config.add_pos_emb:
            input += self.transformer.wpe(pos)
        return input, pos

    def forward(self, input, mask, inputk=None, pos=None, posk=None):
        """
            inputk support is generally not important as multiple layer level
            RetroMAE needs cross attention + single layer hence this
        """
        if mask.dtype != torch.bool:
            mask = (mask == 0)

        # basic transformer execution
        x, pos = self.preprocess_input(input, pos)
        if self.config.project_input > 0:
            x = self.transformer.proj_in(x)

        if inputk is not None:
            xk_global, posk_global = self.preprocess_input(inputk, posk)
        else:
            xk_global, posk_global = None, None

        if not self.config.prelayernorm:
            x = self.transformer.ln(x)
        x = self.transformer.drop(x)
        layerwise = [x]
        for block_num, block in enumerate(self.transformer.h):
            if inputk is not None:
                xk, posk = xk_global, posk_global
            else:
                xk, posk = x, pos

            if self.config.cls_optimize_last_layer and (block_num == (self.config.n_layer-1)):
                x, block_layerwise = block(x[:,:1,:], xk, pos[:,:1], posk, mask)
            else:
                x, block_layerwise = block(x, xk, pos, posk, mask)
            layerwise.extend(block_layerwise)
        if self.config.prelayernorm:
            x = self.transformer.ln(x)

        if self.config.decoder:
            x = self.decoder(x)
        return x, layerwise


class EncoderDecoder(nn.Module):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        if config['type'] == 'retromae':
            self.actual_module = RetroMAEEncoderDecoder(config, encoder, decoder)
        elif config['type'] == 'mae_v1':
            self.actual_module = MAEV1EncoderDecoder(config, encoder, decoder)

    def forward(self, input):
        return self.actual_module(input)


class MAEV1EncoderDecoder(nn.Module):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        if config.get('tie_vocab', False):
            logger.info('Encoder decoder vocab weights tied')
            decoder.vocab_mat().weight = encoder.vocab_mat().weight

        if config.get('tie_pos', False):
            logger.info('Encoder decoder pos embeddings tied')
            decoder.pos_emb().weight = encoder.pos_emb().weight

    def forward(self, input):
        encoder_output = self.encoder(input['xfeat'])
        decoder_feat = input['yfeat']
        bsz, max_hints, num_hints = decoder_feat['hint_input_ids'].shape
        valid_idx = torch.where(decoder_feat['target_mask'].flatten(0,1))[0]

        # encoder_output[:, None, None, :].repeat((1, max_hints, num_hints, 1)).flatten(0,1)[valid_idx].shape
        input_ids = encoder_output[:, None, None, :].repeat((1, max_hints, 1, 1)).flatten(0,1)[valid_idx]
        pos = decoder_feat['target_pos'].flatten(0,1)[valid_idx, None]
        # dict_keys(['hint_input_ids', 'hint_pos', 'target_input_ids', 'target_pos', 'target_mask'])

        # decoder input ids
        input_keys = decoder_feat['hint_input_ids'].flatten(0,1)[valid_idx]
        pos_keys = decoder_feat['hint_pos'].flatten(0,1)[valid_idx]
        # decoder attention mask
        attention_mask = torch.ones_like(input_keys)
        # for keys, provide embeddings directly
        input_keys = self.decoder.get_token_emb(input_keys)
        label = decoder_feat['target_input_ids'].flatten(0,1)[valid_idx]

        decoder_input = {
            'input_ids': input_ids,
            'pos': pos.to(torch.long),
            'attention_mask': attention_mask,
            'input_keys': input_keys,
            'pos_keys': pos_keys.to(torch.long)
        }
        decoder_output = self.decoder(decoder_input)
        return {'logits': decoder_output[:, 0], 'label': label.to(torch.long)}


class RetroMAEEncoderDecoder(nn.Module):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.mask_rate = config.get('mask_rate', 0.5)
        self.directions = config.get('directions', 'xy').split(',')
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

        def valid_direction(direction):
            return len(direction) == 2, all([k in ['x', 'y'] for k in direction])
        assert all(d in ['xy', 'yx', 'xx', 'yy'] for d in self.directions), "direction is xy or yx"

        if config.get('tie_vocab', False):
            logger.info('Encoder decoder vocab weights tied')
            decoder.vocab_mat().weight = encoder.vocab_mat().weight

        if config.get('tie_pos', False):
            logger.info('Encoder decoder pos embeddings tied')
            decoder.pos_emb().weight = encoder.pos_emb().weight

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

    def forward(self, input):
        res, cache = {}, {}
        for direction in self.directions:
            logitname, labelname = f'{direction}logits', f'{direction}label'
            res[logitname], res[labelname], cache = self.forward_direction(input, direction, cache)
        return res
