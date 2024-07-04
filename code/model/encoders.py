import onnxruntime
import logging
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_helpers import init_weights
from transformers import AutoModel, AutoTokenizer, BertConfig
from transformers import DistilBertModel, DistilBertConfig
from sentence_transformers import SentenceTransformer
from model.transformer.arch import Transformer, EncoderDecoder
from model.transformer.me5_adapters import fuse_rope_into_me5, fuse_glu_into_me5


def get_nomic_modules():
    if torch.cuda.get_device_name('cuda:0').startswith('AMD'):
        # AMD still doesn't support rope flash attention, so bound to use non flash version
        from model.transformer.nomicbert.modeling_hf_nomic_bert import NomicBertModel, NomicBertForPreTraining
    else:
        from model.transformer.nomicbert.modeling_nomic_bert import NomicBertModel, NomicBertForPreTraining
    from model.transformer.nomicbert.configuration_nomic_bert import NomicBertConfig
    return NomicBertModel, NomicBertForPreTraining, NomicBertConfig

def get_flash_modules():
    from model.transformer.flashbert.bert import BertModel as FlashBertModel
    return FlashBertModel

def get_mosaic_modules():
    from model.transformer.mosaicbert.bert_layers import BertModel as MosaicBertModel
    return MosaicBertModel


logger = logging.getLogger(__name__)


def create_encoder(enc_conf, encoders):
    if not isinstance(enc_conf, dict) or 'type' not in enc_conf:
        raise NotImplementedError
    enc_type = enc_conf['type']
    if enc_type == 'combiner':
        encoder = Combiner(enc_conf, encoders)
    elif enc_type == 'custom_enc_dec':
        encoder = EncoderDecoder(enc_conf.get('params', {}), encoders[enc_conf['encoder']],
                                 encoders[enc_conf['decoder']])
    elif enc_type == 'sequential':
        encoder = CustomSequential(*[encoders[k] for k in enc_conf['list']])
    else:
        encoder = SingleEncoder(enc_conf)

    # This support would be removed in the future
    # TODO: Remove this support, would not be backward compatible
    if enc_conf.get('projection', None) is not None:
        encoder = CustomSequential(encoder, LinearProjection(enc_conf['projection'], in_dim=encoder.out_dim))

    # # by default compile the model if possible
    # if hasattr(torch, 'compile'):
    #     encoder = torch.compile(encoder)
    compile_flag = enc_conf.get('compile', False)
    if compile_flag:
        logger.info('compiling model')
        if isinstance(compile_flag, str):
            encoder = torch.compile(encoder, mode=compile_flag)
        elif compile_flag is True:
            encoder = torch.compile(encoder)
        else:
            raise NotImplementedError

    load = enc_conf.get('load', None)
    if load is not None:
        sd = torch.load(load['path'], map_location='cpu')
        if not load.get('state_dict_at_root', False):
            sd = sd['model_state_dict']
        if 'transformations' in load:
            for trf in load['transformations']:
                assert isinstance(trf, dict) and len(trf) == 1
                if 'replace_start' in trf:
                    find, replace = trf['replace_start'].split('=>')
                    sd = {(replace+k[len(find):] if k.startswith(find) else k): v for k, v in sd.items()}
                elif 'filter' in trf:
                    sd_filter = {}
                    for pattern in trf['filter'].split('|'):
                        sd_filter.update({k: v for k, v in sd.items() if k.startswith(pattern)})
                    sd = sd_filter
                elif 'remove_start' in trf:
                    sd = {k: v for k, v in sd.items() if not k.startswith(trf['remove_start'])}
                else:
                    raise NotImplementedError
        res = encoder.load_state_dict(sd, strict=load.get('strict', True))
        if not load.get('strict'):
            logger.info(f'missing keys {res.missing_keys}')
            logger.info(f'unexpected keys {res.unexpected_keys}')

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
    def __init__(self, transformer='roberta-base'):
        super(STransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.out_dim = self.transformer.get_sentence_embedding_dimension()

    def encode(self, data):
        return self.transformer(data)['sentence_embedding']

    def forward(self, data):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
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
        weights = weights + ((mask - 1).type(torch.float) / 1e-8).unsqueeze(2).type(weights.dtype)
        weights = F.softmax(weights, dim=1)
        return (term_tensor * weights).sum(dim=1)


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        if isinstance(config, str):
            self.cls_id = None
            self.pooling_mode = config
        elif isinstance(config, dict):
            self.cls_id = config.get('cls_id', None)
            self.pooling_mode = config['mode']
            if self.pooling_mode == 'weighted':
                self.weight = nn.Linear(config['dim'], 1)
            elif self.pooling_mode == 'multihead_weighted':
                self.dim = config['dim']
                self.num_heads = config['num_heads']
                self.per_head_dim = self.dim//self.num_heads
                self.weight = nn.Linear(self.dim, self.num_heads)
        else:
            raise NotImplementedError
        assert self.pooling_mode in ['mean', 'max', 'min', 'cls', 'weighted', 'multihead_weighted', 'all']

    def forward(self, hidden_states, attention_mask, input_ids=None):
        if self.pooling_mode == 'mean':
            return torch.sum(hidden_states * attention_mask.unsqueeze(-1), 1) / torch.clamp(
                attention_mask.sum(1).unsqueeze(-1), min=1e-9)
        elif self.pooling_mode == 'max':
            return torch.max(hidden_states, 1)[0]
        elif self.pooling_mode == 'min':
            return torch.min(hidden_states, 1)[0]
        elif self.pooling_mode == 'cls':
            if self.cls_id is None:
                return hidden_states[:, 0]
            else:
                return hidden_states[torch.where(input_ids==self.cls_id)]
        elif self.pooling_mode == 'weighted':
            weights = self.weight(hidden_states)
            weights = weights + ((attention_mask - 1).type(torch.float) / 1e-8).unsqueeze(-1).type(weights.dtype)
            weights = F.softmax(weights, dim=1)
            return (hidden_states * weights).sum(dim=1)
        elif self.pooling_mode == 'multihead_weighted':
            weights = self.weight(hidden_states)  # [B, T, num_heads]
            hidden_states = hidden_states.view((*hidden_states.shape[:-1], self.num_heads,
                                                self.per_head_dim))  # [B, T, num_heads, per_head_dim]
            weights += ((attention_mask - 1).type(torch.float) / 1e-8).unsqueeze(-1).type(weights.dtype)
            weights = F.softmax(weights, dim=1)  # [B, T, num_heads] normalize along T
            return (hidden_states * weights.unsqueeze(-1)).sum(dim=1).view((-1, self.dim))
        elif self.pooling_mode == 'all':
            # return attention_mask.unsqueeze(2)*hidden_states
            # TODO: check if we need this attention mask multiplication
            return hidden_states
        else:
            raise NotImplementedError


class DistilBertCustomParamsInputLayer(nn.Module):
    def __init__(self, transformer, pooling_config, prune_tok=False):
        super(DistilBertCustomParamsInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = Pooler(pooling_config)
        self.out_dim = self.transformer.config.hidden_size
        self.prune_tok = prune_tok

    def vocab_mat(self):
        return self.transformer.embeddings.word_embeddings

    def get_token_emb(self, tokens):
        return self.transformer.embeddings.word_embeddings(tokens)

    def encode(self, data):
        if self.prune_tok:
            numtok = min(data['attention_mask'].shape[-1], (data['attention_mask'].sum(-1).max() + 15) & (-16))
            data['input_ids'] = data['input_ids'][..., :numtok]
            data['attention_mask'] = data['attention_mask'][..., :numtok]
        else:
            numtok = data['attention_mask'].shape[-1]
        input_ids, attention_mask = data['input_ids'].view([-1, numtok]), data['attention_mask'].view([-1, numtok])
        emb_tokens = self.transformer(input_ids, attention_mask)['last_hidden_state']
        embeddings = self.pooler(emb_tokens, attention_mask)
        return embeddings.view([*data['input_ids'].shape[:-1], embeddings.shape[-1]])

    def forward(self, data):
        # # TODO: doesn't work seamlessly with grad cache, make suitable changes
        # maxlen = data['attention_mask'].sum(1).max()
        # data['input_ids'] = data['input_ids'][:, :maxlen]
        # data['attention_mask'] = data['attention_mask'][:, :maxlen]
        # # TODO: 'mask' portion assumes single embedding per input, need to relax that
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class TwinBertInputLayer(nn.Module):
    def __init__(self, transformer):
        super(TwinBertInputLayer, self).__init__()
        self.transformer = transformer
        self.out_dim = self.transformer.out_dim

    def encode(self, data):
        return self.transformer(seq=data['input_ids'], mask=data['attention_mask'])

    def forward(self, data):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class AutoModelInputLayer(nn.Module):
    def __init__(self, transformer, config):
        super(AutoModelInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = Pooler(config['pooler'])
        self.config = config

    def get_token_emb(self, tokens, *args, **kwargs):
        return self.transformer.embeddings.word_embeddings(tokens)

    def pos_emb(self, *args, **kwargs):
        return self.transformer.embeddings.position_embeddings

    def vocab_mat(self, *args, **kwargs):
        return self.transformer.embeddings.word_embeddings

    @property
    def out_dim(self):
        return self.config['out_dim']

    @property
    def output_key(self):
        return self.config['output_key']

    def encode(self, data):
        return self.pooler(self.transformer(**data)[self.output_key], data['attention_mask'])

    def forward(self, data):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class NomicBert(nn.Module):
    def __init__(self, config={}):
        super(NomicBert, self).__init__()
        NomicBertModel, _, _ = get_nomic_modules()
        self.bert = NomicBertModel.from_pretrained(config['path'], add_pooling_layer=False)

        def override_dropout(model, value):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    if module.p > 0:
                        module.p = value
        self.override_nonzero_dropout = config.get('override_nonzero_dropout', None)
        if self.override_nonzero_dropout is not None:
            logger.info(f'override dropout in bert model to {self.override_nonzero_dropout}')
            override_dropout(self.bert, self.override_nonzero_dropout)

        self.pooler = Pooler(config['pooler'])
        if 'trim_layers' in config:
            self.bert.encoder.layers = nn.Sequential(*list(self.bert.encoder.layers.children())[:-config['trim_layers']])
        self.out_dim = 768

        if config.get('randomize'):
            self.bert.apply(partial(init_weights, std=self.bert.config.initializer_range))

    def get_token_emb(self, tokens, *args, **kwargs):
        raise NotImplementedError

    def pos_emb(self, *args, **kwargs):
        raise NotImplementedError

    def vocab_mat(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, data):
        numtok = data['input_ids'].shape[-1]
        input_ids, attention_mask = data['input_ids'].view([-1, numtok]), data['attention_mask'].view([-1, numtok])
        emb_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = self.pooler(emb_tokens, attention_mask, input_ids=input_ids)
        try:
            embeddings = embeddings.view([*data['input_ids'].shape[:-1], embeddings.shape[-1]])
        except:
            import pdb; pdb.set_trace()
        return embeddings

    def forward(self, data):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class NomicBertRandom(NomicBert):
    '''
        Randomly initialized Nomic model to verify the efficacy of muP
    '''
    def __init__(self, config={}):
        super(NomicBert, self).__init__()
        NomicBertModel, _, NomicBertConfig = get_nomic_modules()
        bert_config = NomicBertConfig.from_pretrained(config['path'])
        self.emb_scale = config['emb_scale']
        assert isinstance(self.emb_scale, int)
        bert_config.n_embd *= self.emb_scale
        bert_config.n_head *= self.emb_scale
        bert_config.n_inner *= self.emb_scale
        self.bert = NomicBertModel(bert_config, add_pooling_layer=False)

        def override_dropout(model, value):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    if module.p > 0:
                        module.p = value
        self.override_nonzero_dropout = config.get('override_nonzero_dropout', None)
        if self.override_nonzero_dropout is not None:
            logger.info(f'override dropout in bert model to {self.override_nonzero_dropout}')
            override_dropout(self.bert, self.override_nonzero_dropout)

        self.pooler = Pooler(config['pooler'])
        if 'trim_layers' in config:
            self.bert.encoder.layers = nn.Sequential(*list(self.bert.encoder.layers.children())[:-config['trim_layers']])
        self.out_dim = bert_config.n_inner

        # apply muP initialization
        self.bert.apply(partial(init_weights,
                                std=self.bert.config.initializer_range/math.sqrt(self.emb_scale),
                                std_emb=self.bert.config.initializer_range))

    def encode(self, data):
        numtok = data['input_ids'].shape[-1]
        input_ids, attention_mask = data['input_ids'].view([-1, numtok]), data['attention_mask'].view([-1, numtok])
        emb_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = self.pooler(emb_tokens, attention_mask)
        # muP output scaling
        embeddings /= self.emb_scale
        return embeddings.view([*data['input_ids'].shape[:-1], embeddings.shape[-1]])


class NomicBertMLM(nn.Module):
    def __init__(self, config={}):
        super(NomicBertMLM, self).__init__()
        _, NomicBertForPreTraining, NomicBertConfig = get_nomic_modules()
        bert_config = NomicBertConfig.from_pretrained(config['path'])
        self.emb_scale = config['emb_scale']
        assert isinstance(self.emb_scale, int)
        bert_config.n_embd *= self.emb_scale
        bert_config.n_head *= self.emb_scale
        bert_config.n_inner *= self.emb_scale
        bert_config.mlm_logit_scale = 1.0/self.emb_scale
        self.bert = NomicBertForPreTraining(bert_config)

        # apply muP initialization overriding inbuild initialization
        self.bert.apply(partial(init_weights,
                                std=self.bert.config.initializer_range/math.sqrt(self.emb_scale),
                                std_emb=self.bert.config.initializer_range))

    def forward(self, data):
        return self.bert(input_ids=data['input_ids'],
                         attention_mask=data['attention_mask'],
                         labels=data['labels'])


class MosaicBert(NomicBert):
    def __init__(self, config={}):
        super(NomicBert, self).__init__()
        MosaicBertModel = get_mosaic_modules()
        self.bert = MosaicBertModel.from_pretrained(config['path'], add_pooling_layer=False)

        def override_dropout(model, value):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    if module.p > 0:
                        module.p = value
        self.override_nonzero_dropout = config.get('override_nonzero_dropout', None)
        if self.override_nonzero_dropout is not None:
            logger.info(f'override dropout in bert model to {self.override_nonzero_dropout}')
            override_dropout(self.bert, self.override_nonzero_dropout)

        self.pooler = Pooler(config['pooler'])
        if 'trim_layers' in config:
            self.bert.encoder.layers = nn.Sequential(*list(self.bert.encoder.layers.children())[:-config['trim_layers']])
        self.out_dim = 768

        if config.get('randomize'):
            self.bert.post_init()

    def encode(self, data):
        numtok = data['input_ids'].shape[-1]
        input_ids, attention_mask = data['input_ids'].view([-1, numtok]), data['attention_mask'].view([-1, numtok])
        emb_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        embeddings = self.pooler(emb_tokens, attention_mask)
        return embeddings.view([*data['input_ids'].shape[:-1], embeddings.shape[-1]])


class FlashBert(nn.Module):
    def __init__(self, config={}):
        super(FlashBert, self).__init__()
        FlashBertModel = get_flash_modules()
        self.bert = FlashBertModel.from_pretrained(config['path'],
                                              config=BertConfig.from_pretrained(config['path']),
                                              add_pooling_layer=False)
        def override_dropout(model, value):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    if module.p > 0:
                        module.p = value
        self.override_nonzero_dropout = config.get('override_nonzero_dropout', None)
        if self.override_nonzero_dropout is not None:
            logger.info(f'override dropout in bert model to {self.override_nonzero_dropout}')
            override_dropout(self.bert, self.override_nonzero_dropout)

        self.pooler = Pooler(config['pooler'])
        if 'trim_layers' in config:
            self.bert.encoder.layers = nn.Sequential(*list(self.bert.encoder.layers.children())[:-config['trim_layers']])
        self.out_dim = config['out_dim']

    def get_token_emb(self, tokens, *args, **kwargs):
        raise NotImplementedError

    def pos_emb(self, *args, **kwargs):
        raise NotImplementedError

    def vocab_mat(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, data):
        numtok = data['input_ids'].shape[-1]
        input_ids, attention_mask = data['input_ids'].view([-1, numtok]), data['attention_mask'].view([-1, numtok])
        emb_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask.bool()).last_hidden_state
        embeddings = self.pooler(emb_tokens, attention_mask)
        return embeddings.view([*data['input_ids'].shape[:-1], embeddings.shape[-1]])

    def forward(self, data):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask] = self.encode(data)
            return emb
        else:
            return self.encode(data)


class CustomTransformerInputLayer(nn.Module):
    def __init__(self, transformer, pooling_config):
        super(CustomTransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = Pooler(pooling_config)
        self.out_dim = self.transformer.out_dim

    def get_token_emb(self, *args, **kwargs):
        return self.transformer.get_token_emb(*args, **kwargs)

    def pos_emb(self, *args, **kwargs):
        return self.transformer.pos_emb(*args, **kwargs)

    def vocab_mat(self, *args, **kwargs):
        return self.transformer.vocab_mat(*args, **kwargs)

    def encode(self, data, return_layerwise=False):
        trf_out, layerwise = self.transformer(data['input_ids'], data['attention_mask'],
                                                 inputk = data.get('input_keys', None),
                                                 pos = data.get('pos', None),
                                                 posk = data.get('pos_keys', None))
        pooled_res = self.pooler(trf_out, data['attention_mask'])
        if return_layerwise:
            layerwise = [self.pooler(layeremb, data['attention_mask']) for layeremb in layerwise]
        return pooled_res, layerwise

    def forward(self, data, return_layerwise=False):
        if 'mask' in data:
            mask = data['mask']
            emb = torch.zeros((data['input_ids'].shape[0], self.out_dim), dtype=torch.float32, device=mask.device)
            if torch.any(torch.any(mask)):
                emb[mask], layerwise = self.encode(data, return_layerwise)
            elif return_layerwise:
                raise NotImplementedError
        else:
            emb, layerwise = self.encode(data, return_layerwise)
        if return_layerwise:
            return emb, layerwise
        else:
            return emb


class E5Model(nn.Module):
    def __init__(self, path, prefix, max_len):
        super(E5Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.prefix = prefix
        self.max_len = max_len

    def get_token_emb(self, *args, **kwargs):
        raise NotImplementedError

    def pos_emb(self, *args, **kwargs):
        raise NotImplementedError

    def vocab_mat(self, *args, **kwargs):
        raise NotImplementedError

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, data, return_layerwise=False):
        assert not return_layerwise
        assert 'mask' not in data
        batch_dict = self.tokenizer([self.prefix+x for x in data['text']],
                                    max_length=self.max_len,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')
        batch_dict = {
            'input_ids': batch_dict['input_ids'].cuda(),
            'attention_mask': batch_dict['attention_mask'].cuda()
        }
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)


class BGEM3(nn.Module):
    def __init__(self, prefix, max_len):
        super(BGEM3, self).__init__()
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.model[0].max_seq_length = max_len
        self.prefix = prefix

    def get_token_emb(self, *args, **kwargs):
        raise NotImplementedError

    def pos_emb(self, *args, **kwargs):
        raise NotImplementedError

    def vocab_mat(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, data, return_layerwise=False):
        assert not return_layerwise
        assert 'mask' not in data
        emb = self.model.encode([self.prefix+x for x in data['text']],
                                normalize_embeddings=True,
                                convert_to_numpy=False,
                                show_progress_bar=False,
                                batch_size=256)
        emb = torch.stack(emb)
        return emb


class Residual(nn.Module):
    def __init__(self, dim, _type):
        super(Residual, self).__init__()
        self.dim = dim
        self.type = _type
        self.mat = nn.Linear(dim, dim)

    def forward(self, inp):
        if self.type == 'relu':
            return inp+F.relu(self.mat(inp))
        elif self.type == 'gelu':
            return inp+F.gelu(self.mat(inp))
        elif self.type == 'pre_relu':
            return inp+self.mat(F.relu(inp))
        elif self.type == 'pre_gelu':
            return inp+self.mat(F.gelu(inp))
        elif self.type == 'relu_norm':
            return F.normalize(inp+F.relu(self.mat(inp)), dim=-1)
        elif self.type == 'gelu_norm':
            return F.normalize(inp+F.gelu(self.mat(inp)), dim=-1)
        elif self.type == 'relu_resnorm':
            return inp+F.normalize(F.relu(self.mat(inp)), dim=-1)
        elif self.type == 'gelu_resnorm':
            return inp+F.normalize(F.gelu(self.mat(inp)), dim=-1)
        else:
            raise NotImplementedError


class LinearProjection(nn.Module):
    def __init__(self, config, in_dim=None):
        super(LinearProjection, self).__init__()

        self.config = config
        self.in_dim, self.out_dim, self.num_layers = in_dim, None, 0
        self.residual_type, self.init, self.normalize_input = 'relu', None, False
        self.add_first_dims = False
        self.first_dims_only = False

        if isinstance(config, int):
            self.out_dim = config
        elif isinstance(config, dict):
            self.in_dim = config.get('in_dim', self.in_dim)
            self.out_dim = config['out_dim']
            self.num_layers = config.get('num_layers', 0)
            self.residual_type = config.get('residual_type', 'relu')
            self.init = config.get('init', None)
            self.normalize_input = config.get('normalize_input', False)
            self.add_first_dims = config.get('add_first_dims', False)
            self.first_dims_only = config.get('first_dims_only', False)
            if self.first_dims_only:
                assert not self.add_first_dims
                assert self.num_layers == 0
        else:
            raise NotImplementedError

        assert self.in_dim is not None and self.out_dim is not None

        modules = [Residual(self.in_dim, self.residual_type) for _ in range(self.num_layers)]
        if not self.first_dims_only:
            modules.append(nn.Linear(self.in_dim, self.out_dim))
        self.linear_layers = nn.Sequential(*modules)

        if self.init is not None:
            def randomize_f(module_):
                return init_weights(module_, self.init['range'], self.init['clamp'])
            self.linear_layers.apply(randomize_f)

    def project(self, embs):
        if self.first_dims_only:
            return embs[..., :self.out_dim]
        proj_embs = self.linear_layers(embs)
        if self.add_first_dims:
            proj_embs += embs[..., :self.out_dim]
        return proj_embs

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            embs, mask = data, None
        elif isinstance(data, dict):
            embs, mask = data['embs'], data.get('mask', None)
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


class EmbNormalize(nn.Module):
    """
    Dummy model for normalizing input embeddings, useful for evaluating baselines
    """
    def __init__(self, config):
        super(EmbNormalize, self).__init__()
        self.config = config

    def forward(self, data):
        embs = F.normalize(data['embs'], dim=-1)
        if 'mask' in data:
            mask = data['mask']
            embs[~mask] = 0
        return embs


class OnnxEncoder(nn.Module):
    """
    Dummy model for normalizing input embeddings, useful for evaluating baselines
    """
    def __init__(self, config):
        super(OnnxEncoder, self).__init__()
        self.config = config
        self.onnx_path = config['onnx_path']
        self.setup = False
        if config.get('device') == 'cpu':
            providers = [('CPUExecutionProvider')]
            sessionOptions = onnxruntime.SessionOptions()
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path, sess_options=sessionOptions,
                                                            providers=providers)
            self.setup = True

    def forward(self, data):
        device = data['input_ids'].device
        if not self.setup:
            providers = [('CUDAExecutionProvider', {'device_id': device.index})]
            sessionOptions = onnxruntime.SessionOptions()
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path, sess_options=sessionOptions,
                                                            providers=providers)
            self.setup = True

        ort_inputs = {
            self.ort_session.get_inputs()[0].name: data['input_ids'].detach().cpu().numpy(),
            self.ort_session.get_inputs()[1].name: data['attention_mask'].detach().cpu().numpy()
        }
        res = self.ort_session.run(None, ort_inputs)
        return torch.from_numpy(res[0]).to(device)


class SingleEncoder(nn.Module):
    def __init__(self, enc_conf):
        super().__init__()
        self.enc_conf = enc_conf
        enc_type = enc_conf['type']
        if enc_type == 'distilbert_customparams':
            emb_model = DistilBertModel(DistilBertConfig(**enc_conf.get('params', {})))
            encoder = DistilBertCustomParamsInputLayer(emb_model, enc_conf.get('pooling_mode', 'mean'), enc_conf.get('prune_tok', False))
            out_dim = encoder.out_dim
        elif enc_type == 'sbert':
            enc_name = enc_conf['desc']
            encoder = STransformerInputLayer(SentenceTransformer(enc_name))
            if enc_conf.get('trim_layers', None):
                if enc_conf.get('roberta', False):
                    trimmed_layers = encoder.transformer[0].auto_model.encoder.layer[:-enc_conf['trim_layers']]
                    encoder.transformer[0].auto_model.encoder.layer = trimmed_layers
                else:
                    trimmed_layers = encoder.transformer[0].auto_model.transformer.layer[:-enc_conf['trim_layers']]
                    encoder.transformer[0].auto_model.transformer.layer = trimmed_layers
            if enc_conf.get('fuse_rope_into_me5'):
                fuse_rope_into_me5(encoder.transformer)
            if enc_conf.get('fuse_glu_into_me5'):  # fuse before randomize intentionally
                fuse_glu_into_me5(encoder.transformer)
            if enc_conf.get('randomize'):
                encoder.apply(partial(init_weights, std=enc_conf['randomize']))
            if enc_conf.get('change_mean_to_cls_pooler'):
                encoder.transformer[1].pooling_mode_mean_tokens = False
                encoder.transformer[1].pooling_mode_cls_token = True
            logger.info(encoder.transformer[1].get_pooling_mode_str())
            out_dim = encoder.transformer.get_sentence_embedding_dimension()
        elif enc_type == 'e5':
            encoder = E5Model(enc_conf['path'], enc_conf['prefix'], enc_conf['max_len'])
            out_dim = enc_conf['out_dim']
        elif enc_type == 'bge_m3':
            encoder = BGEM3(enc_conf['prefix'], enc_conf['max_len'])
            out_dim = enc_conf['out_dim']
        elif enc_type == 'twinbert':
            from model.twinbert.backbone import TwinBertBackbone
            emb_model = TwinBertBackbone(enc_conf)
            out_dim = emb_model.out_dim
            encoder = TwinBertInputLayer(emb_model)
        elif enc_type == 'nomicbert':
            encoder = NomicBert(config=enc_conf.get('config', {}))
            out_dim = encoder.out_dim
        elif enc_type == 'mosaicbert':
            encoder = MosaicBert(config=enc_conf.get('config', {}))
            out_dim = encoder.out_dim
        elif enc_type == 'nomicbert_random':
            encoder = NomicBertRandom(config=enc_conf.get('config', {}))
            out_dim = encoder.out_dim
        elif enc_type == 'nomicbert_mlm':
            encoder = NomicBertMLM(config=enc_conf.get('config', {}))
            out_dim = 1  # directly outputs loss scalar
        elif enc_type == 'flashbert':
            encoder = FlashBert(config=enc_conf.get('config', {}))
            out_dim = encoder.out_dim
        elif enc_type == 'hf_automodel':
            enc_name = enc_conf['desc']
            encoder = AutoModelInputLayer(AutoModel.from_pretrained(enc_name), config=enc_conf['config'])
            out_dim = encoder.out_dim
        elif enc_type == 'custom_transformer':
            transformer = Transformer(enc_conf)
            encoder = CustomTransformerInputLayer(transformer, enc_conf.get('pooling_mode', 'mean'))
            out_dim = encoder.out_dim
        elif enc_type == 'lin_proj':
            encoder = LinearProjection(enc_conf['out_dim'], in_dim=enc_conf['in_dim'])
            out_dim = enc_conf['out_dim']
        elif enc_type == 'normalize':
            encoder = EmbNormalize(enc_conf)
            out_dim = enc_conf['out_dim']
        elif enc_type == 'onnx':
            encoder = OnnxEncoder(enc_conf)
            out_dim = enc_conf['out_dim']
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
        assert config['type'] == 'combiner'

        encoders_list = config['encoders']
        if isinstance(encoders_list, str):
            encoders_list = [x.strip() for x in encoders_list.split(',')]
        elif not isinstance(encoders_list, list):
            raise NotImplementedError
        self.encoders = nn.ModuleList(input_encoders[k] for k in encoders_list)

        if config['combiner_type'] == 'mean':
            self.type = 'mean'
        elif config['combiner_type'] == 'weighted_mean':
            self.type = 'weighted_mean'
            self.weights = torch.tensor(list(map(float, config['weights'].split(','))))[:, None, None]
        elif config['combiner_type'] == 'norm_mean':
            self.type = 'norm_mean'
        elif config['combiner_type'] == 'self_attention':
            self.type = 'self_attention'
            self.attn = nn.MultiheadAttention(config['embed_dim'], config['num_heads'])
        else:
            raise NotImplementedError

    def forward(self, listfeat):
        inp = [encoder(mode_feat) for encoder, mode_feat in zip(self.encoders, listfeat)]
        if self.type == 'mean' or self.type == 'norm_mean' or self.type == 'weighted_mean':
            inp = torch.stack(inp)
            if self.type == 'norm_mean':
                inp = F.normalize(inp, dim=-1)
            elif self.type == 'weighted_mean':
                if self.weights.device != inp.device:
                    self.weights = self.weights.to(inp.device)
                inp = inp*self.weights
            if any(['mask' in feat_mode for feat_mode in listfeat]):
                masks = []
                for feat_mode in listfeat:
                    masks.append(feat_mode.get('mask', torch.ones(len(inp[0]), dtype=torch.bool, device=inp.device)))
                masks = torch.stack(masks)
                return (inp*masks.unsqueeze(2)).sum(0) / masks.sum(0).unsqueeze(1)
            else:
                return inp.mean(0)
        elif self.type == 'self_attention':
            inp = torch.stack(inp)
            masks = []
            for feat_mode in listfeat:
                masks.append(feat_mode.get('mask', torch.ones(len(inp[0]), dtype=torch.bool, device=inp.device)))
            masks = torch.stack(masks)
            attn_emb = self.attn(inp, inp, inp, key_padding_mask=~masks.T, need_weights=False)[0]
            return (attn_emb*masks.unsqueeze(2)).sum(0) / masks.sum(0).unsqueeze(1)
        else:
            raise NotImplementedError
