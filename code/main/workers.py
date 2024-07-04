import torch._dynamo
# temporary fix until we move to pytorch 2.3
# https://github.com/pytorch/pytorch/issues/111636#issuecomment-1869471021
torch._dynamo.config.optimize_ddp = False
import os
import logging
import datetime
import time
import tqdm.autonotebook as tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import wandb
import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from main.main_utils import all_gather

from data.dl import create_dl
from evaluator import Evaluator
from main.forward_backward import ForwardBackward, DeepSpeedForwardBackward


logger = logging.getLogger(__name__)


class WandBHandler():
    """
    Handle all interactions with wandb API inside this class to better control
    Allows lazy mode which can be used to log everything together in the end
    """
    def __init__(self, config, out_dir):
        self.config = config
        self.log_config = config.get('log_config', config)
        self.out_dir = out_dir
        self.lazy = config.get('lazy', True)

        wandb_id_f = os.path.join(self.out_dir, 'wandb_id.txt')
        if os.path.exists(wandb_id_f) and not self.config.get('create_new_key', False):
            self.wandb_id = open(wandb_id_f).read().strip()
        else:
            self.wandb_id = wandb.util.generate_id()
            with open(wandb_id_f, 'w') as fout:
                fout.write(self.wandb_id)

        if not self.lazy:
            self.login_init()
        else:
            self.log_data = []

        if config.get('watch_model', False):
            assert not self.lazy, "watch model not supported in lazy mode"
            wandb.watch(self.model)

    def login_init(self):
        config = self.config
        wandb.login(key=config['api_key'])
        wandb.init(project=config['project'],
                    entity=config['entity'],
                    config=self.log_config,
                    name=config['name'],
                    tags=config['tags'].split('/') if 'tags' in config else None,
                    id=self.wandb_id, resume="allow")

    def log(self, data, step=None):
        if self.lazy:
            self.log_data.append((data, step))
        else:
            wandb.log(data, step=step)

    def finish(self):
        if self.lazy:
            # upload all the logs now in one shot
            logger.info(f'wandb logging start')
            self.login_init()
            for data, step in self.log_data:
                wandb.log(data, step=step)
            wandb.finish()
            logger.info(f'wandb logging end')
        else:
            wandb.finish()


class MLFlowHandler():
    """
    Handle all interactions with wandb API inside this class to better control
    Allows lazy mode which can be used to log everything together in the end
    """
    def __init__(self, config):
        if mlflow.active_run() is not None:
            self.run = mlflow.active_run()
        else:
            self.run = mlflow.start_run()
        self.client = MlflowClient()
        self.config = config
        self.last_logged = time.time()
        self.log_every_seconds = config.get('log_every_seconds', 60)
        self.metrics_cache = []
    def log(self, name, metric, step):
        time_now = time.time()
        self.metrics_cache.append(Metric(key=name, value=metric, timestamp=int(time_now * 1000), step=step))
        if (time_now - self.last_logged) > self.log_every_seconds:
            logger.info('logging into mlflow')
            self.last_logged = time_now
            self.client.log_batch(self.run.info.run_id,  metrics=self.metrics_cache)
            self.metrics_cache = []


class Worker():
    def __init__(self, model, config, rank, world_size, device, *args, **kwargs):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.local_world_size = kwargs.get('local_world_size', self.world_size)
        self.device = device
        self.model = model

        self.out_dir = config['output_dir']
        self.eval_out_dir = os.path.join(self.out_dir, 'eval')
        self.model_out_dir = os.path.join(self.out_dir, 'models')
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.eval_out_dir, exist_ok=True)
        os.makedirs(self.model_out_dir, exist_ok=True)

        self.tb_writer = None
        if 'tensorboard' in config and self.rank == 0:
            out_dir = config['tensorboard']['out_dir']
            if out_dir == "singularity":
                out_dir = f'{os.environ.get("TENSORBOARD_LOG_DIR")}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            logger.info(f'starting tensorboard at {out_dir}')
            self.tb_writer = SummaryWriter(out_dir,
                                           flush_secs=config['tensorboard'].get('flush_secs', 120))
            self.tb_write_loss_scalars = config['tensorboard'].get('write_loss_scalars', False)

        self.wandb_h = None
        if 'wandb' in config and self.rank == 0:
            self.wandb_h = WandBHandler(config['wandb'], self.out_dir)
            self.wandb_logging_freq = config['wandb'].get('logging_frequency', 1)
            self.wandb_write_loss_scalars = config['wandb'].get('write_loss_scalars', False)
            self.wandb_log_scheduler_lr = config['wandb'].get('log_scheduler_lr', False)
            self.wandb_log_model_stats = config['wandb'].get('log_model_stats', False)
            self.wandb_log_dataset_idx = config['wandb'].get('log_dataset_idx', False)

        self.mlflow_h = None
        if 'mlflow' in config and self.rank == 0:
            self.mlflow_h = MLFlowHandler(config['mlflow'])
            self.mlflow_logging_freq = config['mlflow'].get('logging_frequency', 1)

        self.dl, self.forward_backward = None, None
        self.num_data, self.num_batch = 0, -1
        if 'train' in config:
            self.dl = create_dl(config['train']['data'], rank, world_size, device,
                                local_rank=self.local_rank,
                                local_world_size=self.local_world_size)
            if config['train']['forward_backward'].get('deepspeed', False):
                self.forward_backward = DeepSpeedForwardBackward(config['train']['forward_backward'], self.model,
                                                                 self.rank, self.world_size, self.device,
                                                                 train_dl=self.dl)
            else:
                self.forward_backward = ForwardBackward(config['train']['forward_backward'], self.model,
                                                        self.rank, self.world_size, self.device,
                                                        train_dl=self.dl)
            self.save_interval = self.dl.parse_steps(config['train'].get('save_interval', 0))

        self.evaluator = None
        if 'eval' in config:
            model_to_eval = self.model
            # use ema model to evaluate whenever available
            if self.forward_backward is not None and self.forward_backward.ema is not None:
                logger.info('using ema model for evaluation')
                model_to_eval = self.forward_backward.ema.ema_model
            self.evaluator = Evaluator(config['eval'], model_to_eval, self.tb_writer, self.wandb_h, self.mlflow_h,
                                       self.eval_out_dir, rank, world_size, device, self.forward_backward)

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd['model_state_dict'])
        if self.forward_backward:
            self.forward_backward.load_state_dict(sd['forward_backward_state_dict'])
        if self.dl:
            self.dl.load_state_dict(sd['data_state_dict'][self.rank])
        self.num_data, self.num_batch = sd['worker_state_dict']['num_data'], sd['worker_state_dict']['num_batch']
        torch.set_rng_state(sd['rng_state_dict'][self.rank]['torch'].cpu())

    def state_dict(self):
        """
        Needs to be called on all processes
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'forward_backward_state_dict': self.forward_backward.state_dict(),
            'data_state_dict': all_gather(self.dl.state_dict(), self.world_size),
            'worker_state_dict': {'num_data': self.num_data, 'num_batch': self.num_batch},
            'rng_state_dict': all_gather({'torch': torch.get_rng_state()}, self.world_size)
        }

    def train(self):
        # restore from latest checkpoint if cont_after_preemption is True
        if self.config['train'].get('cont_after_preemption', False) and os.path.exists(self.model_out_dir):
            files = os.listdir(self.model_out_dir)
            steps = [int(x[len('step'):-len('.pt')]) for x in files if x.startswith('step') and x.endswith('.pt')]
            if len(steps) > 0:
                load_path = os.path.join(self.model_out_dir, f'step{max(steps)}.pt')
                logger.info(f'load from {load_path}')
                self.load_state_dict(torch.load(load_path, map_location='cpu'))

        self.model.train()
        pbar = tqdm.tqdm(self.dl, desc="progress => ", disable=(self.rank != 0), initial=self.num_batch+1)

        for self.num_batch, batch_data in enumerate(pbar, start=self.num_batch+1):
            self.num_data += batch_data['batch_size'] * self.world_size
            loss, model_stats = self.forward_backward(batch_data)
            pbar.set_description(f"progress => batch: {self.num_batch}/{len(self.dl)} loss: {loss:.5f}")
            if self.tb_writer:
                self.tb_writer.add_scalar('loss/train', loss, self.num_data)
                if self.tb_write_loss_scalars:
                    for k, v in self.model.loss.named_parameters():
                        if v.requires_grad and v.numel() == 1:
                            self.tb_writer.add_scalar(f'loss_params/{k}', v.item(), self.num_data)
            if (self.wandb_h is not None) and (model_stats is not None) and self.wandb_log_model_stats:
                self.wandb_h.log(model_stats)
            if (self.wandb_h is not None) and (self.num_batch+1) % self.wandb_logging_freq == 0:
                self.wandb_h.log({
                    'loss/train': loss,
                    'num_data': self.num_data
                }, step=self.num_batch)
                if self.wandb_log_dataset_idx:
                    self.wandb_h.log({
                    'dataset_idx': batch_data['dataset_idx']
                    }, step=self.num_batch)
                if self.wandb_write_loss_scalars:
                    for k, v in self.model.loss.named_parameters():
                        if v.requires_grad and v.numel() == 1:
                            self.wandb_h.log({f'loss_params/{k}': v.item()}, step=self.num_batch)
                if self.wandb_log_scheduler_lr and self.forward_backward.scheduler is not None:
                    self.wandb_h.log({'lr': self.forward_backward.scheduler.get_last_lr()[0]}, step=self.num_batch)
            if (self.mlflow_h is not None) and (self.num_batch+1) % self.mlflow_logging_freq == 0:
                self.mlflow_h.log('loss/train', loss, self.num_batch)
                self.mlflow_h.log('num_data', self.num_data, self.num_batch)
            if self.save_interval > 0 and (self.num_batch+1) % self.save_interval == 0:
                self.forward_backward.eval()
                sd = self.state_dict()
                if self.rank == 0:
                    torch.save(sd, os.path.join(self.model_out_dir, f'step{self.num_batch}.pt'))
                    if self.forward_backward.ema is not None:
                        sd['model_state_dict'] = self.forward_backward.ema.ema_model.state_dict()
                        torch.save(sd, os.path.join(self.model_out_dir, f'ema{self.num_batch}.pt'))
                self.forward_backward.train()
            if self.evaluator is not None:
                self.evaluator(self.num_batch, self.num_data)

        self.forward_backward.eval()

        if self.save_interval > 0:
            sd = self.state_dict()
            if self.rank == 0:
                torch.save(sd, os.path.join(self.model_out_dir, f'step_fin.pt'))
                if self.forward_backward.ema is not None:
                    sd['model_state_dict'] = self.forward_backward.ema.ema_model.state_dict()
                    torch.save(sd, os.path.join(self.model_out_dir, f'ema_fin.pt'))

    def run(self):
        if self.config.get('load_path', None):
            logger.info(f"Loading model from {self.config['load_path']}")
            self.load_state_dict(torch.load(self.config['load_path'], map_location='cpu'))

        # if model_same_across_devices(self.model, self.rank, self.world_size):
        #     logger.info('model same across devices')
        # else:
        #     raise Exception('model different across devices')

        if 'train' in self.config:
            self.train()

        if self.evaluator is not None:
            self.evaluator(self.num_batch+1, self.num_data, force=True)

        if self.tb_writer:
            # unflushed metrics will be lost if process ends before flush
            self.tb_writer.close()

        if (self.wandb_h is not None):
            self.wandb_h.finish()

    def join(self):
        if self.dl is None:
            return
        if hasattr(self.dl, 'join'):
            self.dl.join()
