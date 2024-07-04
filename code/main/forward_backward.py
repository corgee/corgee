import math
import re
import logging
import torch
import transformers
# import deepspeed
from functools import partial
from contextlib import nullcontext
from torch.utils.checkpoint import set_device_states
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from main.main_utils import move_to_device, split_by_max_batch_size
from main.ema import EMA


logger = logging.getLogger(__name__)


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .val() be called before the
    beginning of training and evaluation respectively.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self,
                 params, 
                 lr=0.0025, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 warmup_steps=0,
                 r=0.0,
                 weight_lr_power=2.0,
                 ):

        logger.info(f'beta1 {betas[0]} beta2 {betas[1]} warmup {warmup_steps}')

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode = True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1-1/beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1-beta1)
                group['train_mode'] = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight/weight_sum
            if not group['train_mode']:
                raise Exception("Not in train mode!")
            for p in group['params']:
                if p.grad is None:
                    continue
                y = p.data # Notation to match theory
                grad = p.grad.data
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(y)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                z = state['z']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                # Reuse grad buffer for memory efficiency
                grad_normalized = grad.div_(denom)
                # Weight decay calculated at y
                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)
                # These operations update y in-place,
                # without computing x explicitly.
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))
                # z step
                z.sub_(grad_normalized, alpha=lr)
            group['k'] = k+1
        return loss


def create_optimizer(config, model):
    optim_args = {'eps': config['eps']}
    if 'no_decay_list' in config:
        no_decay = config['no_decay_list'].split(',')
    else:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config['lr']
        }
    ]
    if 'scale_linears' not in config:
        optimizer_grouped_parameters.append({
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config['weight_decay'],
            'lr': config['lr']
        })
    else:
        # experimental feature for applying muP, only tested with nomic
        scale_down_p_names = config['linear_list'].split(',')
        scale_down_ps, no_scale_down_ps = [], []
        for n, p in param_optimizer:
            if any(nd in n for nd in no_decay):
                continue
            if any(nd in n for nd in scale_down_p_names):
                scale_down_ps.append(p)
            else:
                no_scale_down_ps.append(p)
        optimizer_grouped_parameters.append({
            'params': no_scale_down_ps,
            'weight_decay': config['weight_decay'],
            'lr': config['lr']
        })
        optimizer_grouped_parameters.append({
            'params': scale_down_ps,
            'weight_decay': config['weight_decay'],
            'lr': config['lr']*config['scale_linears']
        })

    if config['type'] == 'adamw':
        # TODO: using fused adam causes loss=nan, debug and fix
        # use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        # optim_args.update(dict(fused=True) if use_fused else dict())
        optim_args['betas'] = (config.get('beta1', 0.9), config.get('beta2', 0.999))
        return torch.optim.AdamW(optimizer_grouped_parameters, **optim_args)
    elif config['type'] == 'zero_adamw':
        optim_args['betas'] = (config.get('beta1', 0.9), config.get('beta2', 0.999))
        return ZeroRedundancyOptimizer(optimizer_grouped_parameters,
                                                               optimizer_class=torch.optim.AdamW,
                                                               **optim_args)
    elif config['type'] == 'adamw_schedule_free':
        optim_args['betas'] = (config.get('beta1', 0.9), config.get('beta2', 0.999))
        optim_args['warmup_steps'] = config.get('warmup_steps', 0)
        return AdamWScheduleFree(optimizer_grouped_parameters, **optim_args)
    else:
        raise NotImplementedError


def get_cyclic_linear_scheduler(optimizer, num_training_steps, num_cycles=1, last_epoch=-1):
    def lr_lambda(current_step):
        cycle_pos = (current_step/num_training_steps)*num_cycles
        lr_frac = cycle_pos-int(cycle_pos)
        lr_frac = 2-2*lr_frac if lr_frac > 0.5 else 2*lr_frac
        return lr_frac
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


# added in pytorch 2.1
def get_inverse_sqrt_schedule(
    optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_scheduler(config, optimizer, train_dl):
    """
    Create a learning rate scheduler.
    Based on: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
    """
    scheduler = config['type'].lower()
    warmup_steps = train_dl.parse_steps(config.get('warmup_steps', 0))
    t_total = len(train_dl)
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmupinvsqrt':
        return get_inverse_sqrt_schedule(optimizer, num_warmup_steps=warmup_steps,
                                                      timescale=config.get('timescale', None))
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        num_cycles = config.get('num_cycles', 0.5)
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total,
                                                            num_cycles=num_cycles)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                               num_warmup_steps=warmup_steps,
                                                                               num_training_steps=t_total)
    elif scheduler == 'cycliclinear':
        num_cycles = config.get('num_cycles', 1)
        return get_cyclic_linear_scheduler(optimizer,
                                           num_training_steps=t_total,
                                           num_cycles=num_cycles)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


class RandContext:
    def __init__(self, devices):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = [], []
        for device in devices:
            with torch.cuda.device(device):
                self.fwd_gpu_devices.append(device)
                self.fwd_gpu_states.append(torch.cuda.get_rng_state())

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class Forward():
    """
    Forward pass only helper. This is needed for loss based evaluation.
    """
    def __init__(self, config, model, rank, world_size, device):
        self.config = config
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.fp16 = config.get('fp16', False)
        self.bf16 = config.get('bf16', False)
        assert not (self.fp16 and self.model.bf16), "either fp16 or bf16 for training"
        # assert self.bf16 or self.fp16, "you can either use fp16 format or bf16 format for training models"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        if self.fp16:
            self.fwd_ctx = torch.cuda.amp.autocast()
        elif self.bf16:
            self.fwd_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.fwd_ctx = nullcontext()
        self.max_forward_batch_size = config.get('max_forward_batch_size', -1)

    def __call__(self, batch_data, return_layerwise=False):
        batch_data = move_to_device(batch_data, self.device)
        with torch.no_grad(), self.fwd_ctx:
            if self.max_forward_batch_size > 0 and batch_data['batch_size'] > self.max_forward_batch_size:
                batch_feats_split = split_by_max_batch_size(batch_data['feats'], self.max_forward_batch_size)
                reps = [self.model(x, return_layerwise=return_layerwise) for x in batch_feats_split]
                reps_cat = {k: torch.cat([d[k] for d in reps]).detach() for k in reps[0].keys()}
            else:
                reps_cat = self.model(batch_data['feats'], return_layerwise=return_layerwise)
            loss = self.model.loss(reps_cat, batch_data['target'])
        return {k: v.item() for k, v in loss.items()} if isinstance(loss, dict) else loss.item()


def get_param_stats(regex_name, regex, mean_std):
    matches = [x for x in mean_std if re.match(regex, x[0])]
    total_len = sum(x[3] for x in matches)
    grad_sq = math.sqrt(sum((x[1][0].item()**2)*x[3] for x in matches) / total_len)
    param_sq = math.sqrt(sum((x[2][0].item()**2)*x[3] for x in matches) / total_len)
    res = {
        f'grad/mean{regex_name}': sum(x[1][1].item()*x[3] for x in matches) / total_len,
        f'grad/std{regex_name}': grad_sq,
        f'param/mean{regex_name}': sum(x[2][1].item()*x[3] for x in matches) / total_len,
        f'param/std{regex_name}': param_sq,
        f'paramnorm_over_gradnorm/{regex_name}': param_sq/max(1e-4, grad_sq)
    }
    return res


def get_model_stats(regexes, named_parameters):
    mean_std = [(n, (torch.std_mean(x.grad, unbiased=False)), (torch.std_mean(x, unbiased=False)), x.numel())
                for n, x in named_parameters if x.grad is not None]
    stats = {}
    for regex_name, regex in regexes.items():
        stats.update(get_param_stats(regex_name, regex, mean_std))
    return stats


class ForwardBackward(Forward):
    def __init__(self, config, model, rank, world_size, device, train_dl=None):
        super().__init__(config, model, rank, world_size, device)
        self.optimizer, self.scheduler, self.ema = None, None, None
        if 'optimizer' in config:
            self.optimizer = create_optimizer(config['optimizer'], model)
            self.load_optim = config.get('load_optim', None)
        if 'scheduler' in config:
            assert self.optimizer is not None and train_dl is not None
            self.scheduler = create_scheduler(config['scheduler'], self.optimizer, train_dl)
        if 'ema' in config:
            assert self.optimizer is not None and train_dl is not None
            ema_config = config['ema']
            self.ema = EMA(
                model,
                beta = ema_config.get('beta', 0.9999),              # exponential moving average factor
                update_after_step = ema_config.get('update_after_step', 100),    # only after this number of .update() calls will it start updating
                update_every = ema_config.get('update_every', 10),          # how often to actually update, to save on compute (updates every 10th .update() call)
            )
        self.grad_clip_norm = config.get('grad_clip_norm', None)
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = float(self.grad_clip_norm)
        self.grad_accumulate_num_steps = config.get('grad_accumulate_num_steps', 1)
        self.model_stats_freq = config.get('model_stats_freq', 0)
        if self.model_stats_freq > 0:
            self.param_bucket_regex = config['param_bucket_regex']
            assert self.model_stats_freq % self.grad_accumulate_num_steps == 0
        self.num_steps = 0
        self.clip_loss_logits = config.get('clip_loss_logits', False)

        if 'emb_keys' in config:
            self.emb_keys = config['emb_keys'].split(',')
            self.emb_dim = config['emb_dim']
        else:
            self.emb_keys = None

    def train(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.train()

    def eval(self):
        if isinstance(self.optimizer, AdamWScheduleFree):
            self.optimizer.eval()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict['scaler'])
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def __call__(self, batch_data):
        self.num_steps += 1
        batch_data = move_to_device(batch_data, self.device)
        if self.max_forward_batch_size > 0 and batch_data['batch_size'] > self.max_forward_batch_size:
            # split features into sub-batches
            batch_feats_split = split_by_max_batch_size(batch_data['feats'], self.max_forward_batch_size)

            # # for using torch.compile, otherwise it compiles in no grad mode
            # if self.num_steps == 1:
            #     with self.fwd_ctx:
            #         warmup_output = self.model(batch_feats_split[0])
            #         logger.info(f'warmup for torch compile done. output type {type(warmup_output)}')
            #         del warmup_output

            if self.emb_keys is not None:
            # TODO: make this cleaner
                # graph-less forward with preallocated memory
                rnd_states = []
                reps = {k: torch.empty((batch_data['batch_size'], self.emb_dim), dtype=torch.float32,
                                    device='cuda').requires_grad_() for k in self.emb_keys}
                with torch.no_grad(), self.fwd_ctx:
                    emb_start = 0
                    for x in batch_feats_split:
                        rnd_states.append(RandContext([self.device]))
                        for k, v in self.model(x).items():
                            # print(v.dtype, v.shape)
                            reps[k][emb_start: emb_start+self.max_forward_batch_size] = v.detach()
                        emb_start += self.max_forward_batch_size
                reps_cat = reps
            else:
                # graph-less forward
                rnd_states, reps = [], []
                with torch.no_grad(), self.fwd_ctx:
                    for x in batch_feats_split:
                        rnd_states.append(RandContext([self.device]))
                        reps.append(self.model(x))
                # concatenate all sub-batch representations
                reps_cat = {k: torch.cat([d[k] for d in reps]).detach().requires_grad_() for k in reps[0].keys()}

            # compute full batch loss and partial gradients
            with self.fwd_ctx:
                loss = self.model.loss(reps_cat, batch_data['target'])
            self.scaler.scale(loss).backward()

            # build gradient cache
            grads = split_by_max_batch_size({k: v.grad for k, v in reps_cat.items() if v.grad is not None},
                                            self.max_forward_batch_size)
            sync_contexts = [self.model.no_sync for _ in range(len(batch_feats_split) - 1)] + [nullcontext]

            # compute sub-batch losses and gradients
            for x, rnd_state, grad, sync_context in zip(batch_feats_split, rnd_states, grads, sync_contexts):
                with sync_context(), rnd_state, self.fwd_ctx:
                    y = self.model({k: x[k] for k in grad})
                    surrogate = sum(torch.dot(y[k].flatten(), grad[k].flatten()) for k in grad)
                    surrogate.backward()
        else:
            with self.fwd_ctx:
                loss = self.model.loss(self.model(batch_data['feats']), batch_data['target'])
            loss = loss / self.grad_accumulate_num_steps
            self.scaler.scale(loss).backward()

        if self.load_optim is not None and self.num_steps == 1:
            # load_optim only works with Adam/ AdamW and only loads moments
            self.optimizer.step()
            current_sd = self.optimizer.state_dict()
            sd = torch.load(self.load_optim, map_location='cpu')['forward_backward_state_dict']['optimizer']
            for var_name in sd['state']:
                exp_avg = sd['state'][var_name].get('exp_avg', None)
                exp_avg_sq = sd['state'][var_name].get('exp_avg_sq', None)
                if exp_avg is not None and exp_avg_sq is not None:
                    current_sd['state'][var_name]['exp_avg'] = exp_avg
                    current_sd['state'][var_name]['exp_avg_sq'] = exp_avg_sq
            self.optimizer.load_state_dict(current_sd)
            logger.info(f'loading optimizer state dict from {self.load_optim}')

        model_stats = None
        if self.num_steps % self.grad_accumulate_num_steps == 0:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if self.model_stats_freq > 0 and self.num_steps % self.model_stats_freq == 0:
                if self.grad_clip_norm is None:  # unscale gradient if not done already
                    self.scaler.unscale_(self.optimizer)
                model_stats = get_model_stats(self.param_bucket_regex, self.model.named_parameters())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if self.ema:
            self.ema.update()

        if self.scheduler:
            self.scheduler.step()

        if self.clip_loss_logits:
            if self.world_size > 1:
                self.model.loss.module.clip_logits()
            else:
                self.model.loss.clip_logits()

        return loss.item(), model_stats


class DeepSpeedForwardBackward():
    def __init__(self, config, model, rank, world_size, device, train_dl=None):
        raise NotImplementedError
        # encoder = model.encoders[config['encoder_name']]
        # self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
        #     args=config['ds_config'],
        #     model=encoder,
        #     model_parameters=encoder.parameters())

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def __call__(self, batch_data):
        raise NotImplementedError
        # loss = self.model_engine(batch_data)
        # self.model_engine.backward(loss)
        # self.model_engine.step()
