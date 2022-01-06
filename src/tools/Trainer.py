
import os
import socket
import sys
from collections import OrderedDict
import math
import logging
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist 
from tqdm import tqdm
from prettytable import PrettyTable
try:
    from apex import amp
except:
    pass

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class Trainer:
    '''
    Overwrite:
    Must:
        forward_step()
    Optional:
        epoch_step()
        metrics_summary()
        optimizer_init()
    '''

    def __init__(self, **kwargs):
        """Basic Trainer
        Args:
            data_config={'dataset':None, 'train_config':{}, 'val_config':{}},
            train_config = {'model':(None, {}), 'loss_module':(None, {}) 'optimizer':(None, {}), 'scheduler':(None, {})},
            record_config = {'tensorboard':None, 'tag':'model', 'class_tag':[], 'metrics_tag':[], 'metrics_key':[], 'record_path':None, 'log_path':None, 'sort_key':None},
            other_config = {'epochs':, 'batch_size':, 'workers':, 'eval_T':, 'warmingup_T':,'pretrian_path':None, 'init_port':None, 'amp_train':False}
        """

        self.summaryer = kwargs['record_config'].get('tensorboard', None)
        self.model_tag = kwargs['record_config']['tag']
        self.record_path = kwargs['record_config']['record_path']
        self.class_tag = kwargs['record_config']['class_tag']
        self.log_path = kwargs['record_config']['log_path']
        self.log_path = os.path.join(self.log_path, self.model_tag + '.log')
        self.metrics_tag = kwargs['record_config'].get('metrics_tag', [])
        self.metrics_key = kwargs['record_config'].get('metrics_key', (['loss'] + self.metrics_tag))
        self.sort_key = kwargs['record_config'].get('sort_key', self.metrics_key[0])
        self.sort_list = {'Train':[], 'Val':[]}

        self.dataset = kwargs['data_config']['dataset']
        self.train_data_config = kwargs['data_config']['train_config']
        self.val_data_config = kwargs['data_config'].get('val_config', None)

        self.model, self.model_config = kwargs['train_config']['model']
        self.loss_module, self.loss_config = kwargs['train_config']['loss_module']
        self.optimizer, self.optimizer_config = kwargs['train_config']['optimizer']
        self.lr = self.optimizer_config['lr']
        self.scheduler, self.scheduler_config = kwargs['train_config'].get('scheduler', (None, None))

        self.epochs = kwargs['other_config']['epochs']
        self.eval_T = kwargs['other_config'].get('eval_T', 1)
        self.warmingup_T = kwargs['other_config'].get('warmingup_T', 0)
        self.workers = kwargs['other_config'].get('workers', 8)
        self.batch_size = kwargs['other_config']['batch_size']
        self.pretrain_path = kwargs['other_config'].get('pretrian_path', None)
        s = socket.socket()
        s.bind(('',0))
        self.init_port = kwargs['other_config'].get('init_port', s.getsockname()[1])
        s.close()
        
        if kwargs['other_config'].get('amp_train', False) and ('apex' in sys.modules):
            self.use_amp = True
        else:
            self.use_amp = False

        self.world_size = torch.cuda.device_count()
        self.random_seed = random.randint(0, 10000)
        
        self._recorder_init()
        self._dataset_init()
        self._PrintConfig()

    def train(self):
        mp.spawn(self._train_procedure, nprocs=torch.cuda.device_count())

    def _train_procedure(self, process_id):
        logger = self._logger_init(process_id)
        rank = process_id
        world_size = self.world_size
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(self.init_port), world_size=torch.cuda.device_count(), rank=process_id)
        
        torch.backends.cudnn.benchmark = True 
        torch.manual_seed(self.random_seed)
        torch.cuda.set_device(process_id)

        model = self.model(**self.model_config)

        if self.pretrain_path != None:
            param = self._load_params(process_id)
            model.load_state_dict(param)

        model.cuda(process_id)

        loss_module = self.loss_module(**self.loss_config)
        loss_module.cuda(process_id)

        if self.warmingup_T == 0:     
            optimizer, scheduler = self.optimizer_init(model, self.lr)
        else:
            optimizer, scheduler = self.optimizer_init(model, self.lr / 10)
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        
        ############################################
        #AMP Initialization
        ############################################

        if self.use_amp:
            model, optimizer = amp.initialize(model, optimizer, min_loss_scale=2.**16)

        model = nn.parallel.DistributedDataParallel(model, device_ids=[process_id])

        # train_dataset, val_dataset = self._dataset_init()
        workers = math.ceil(self.workers / float(world_size))
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
        if hasattr(train_sampler, 'shuffle'):
            train_sampler.shuffle = True
        # train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
        #                                        batch_size=self.batch_size,
        #                                        shuffle=False,
        #                                        num_workers=workers,
        #                                        pin_memory=True,
        #                                        sampler=train_sampler)
        train_loader = FastDataLoader(dataset=self.train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False,
                                               num_workers=workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

        if self.val_dataset != None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)
            if hasattr(val_sampler, 'shuffle'):
                val_sampler.shuffle = False
            # val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
            #                                     batch_size=self.batch_size,
            #                                     shuffle=False,
            #                                     num_workers=workers,
            #                                     pin_memory=True,
            #                                     sampler=val_sampler)
            val_loader = FastDataLoader(dataset=self.val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=workers,
                                                pin_memory=True,
                                                sampler=val_sampler)

        logger.warning('GPU:{} ready for training.'.format(rank))
        dist.barrier()

        if process_id == 0:
            if self.warmingup_T > 0:
                logger.warning('Warming up start.')

        warmingup_flag = True
        for epoch in range(self.epochs):
            if warmingup_flag and epoch >= self.warmingup_T:
                warmingup_flag = False
                # optimizer, scheduler = self.optimizer_init(model, self.lr)
                optimizer.param_groups[0]['lr'] = self.lr
                optimizer.param_groups[0]['initial_lr'] = self.lr
                if self.scheduler is not None:
                    scheduler = self.scheduler(optimizer, **self.scheduler_config) 
                if process_id == 0:
                    logger.warning('Warming up end.')

            train_loader.sampler.set_epoch(epoch)
            if process_id == 0:
                tloader = tqdm(train_loader, desc='{}:{}_Epoch:{:03d}'.format(self.model_tag, 'Train', epoch), ncols=0)
            else:
                tloader = train_loader

            '''
            Train Step
            '''
            # logger.debug(model.module.vnet.attention_four.spatialatt._modules['3'].weight)
            self.epoch_step(process_id, epoch, 'Train', logger, tloader, model, loss_module, optimizer, scheduler)
            '''
            Val Step
            '''
            if self.val_dataset != None and (epoch % self.eval_T) == 0 and epoch != 0:
                if process_id == 0:
                    vloader = tqdm(val_loader, desc='{}:{}_Epoch:{:03d}'.format(self.model_tag, 'Val', epoch), ncols=0)
                else:
                    vloader = val_loader

                with torch.no_grad():
                    self.epoch_step(process_id, epoch, 'Val', logger, vloader, model, loss_module)
            
            if process_id == 0 and (epoch % self.eval_T) == 0:
                params = model.module.state_dict()
                result_dict = {
                    'param': params,
                    'record': self.recorder,
                }
                torch.save(result_dict, os.path.join(self.record_path, '{}_{}.pth'.format(self.model_tag, epoch)))

        if process_id == 0:
            if self.val_dataset != None:
                sorted_list = sorted(self.sort_list['Val'], key=lambda x: x[1], reverse=True)[0:3]
            else:
                sorted_list = sorted(self.sort_list['Train'], key=lambda x: x[1], reverse=True)[0:3]

            best_model_id = sorted_list[0][0]
            best_result = torch.load(os.path.join(self.record_path, '{}_{}.pth'.format(self.model_tag, best_model_id)))
            torch.save(best_result, os.path.join(self.record_path, 'best_model.pth'))
            logger.info(sorted_list)

    def _load_params(self, process_id):
        result = torch.load(self.pretrain_path, map_location='cuda:{}'.format(process_id))
        params = result['params']

        return params

    def epoch_step(self, process_id, epoch, mode, logger, loader, model, loss_module, optimizer=None, scheduler=None):
        if process_id == 0:
            metrics_list = []
            for k, v in self.window_dict.items():
                v.reset()

        if mode == 'Train':
            model.train()
        elif mode == 'Val':
            model.eval()

        for input_data in loader:

            logger.debug('{}_Epoch_{}:{}'.format(mode, epoch, str(input_data[0])))

            output = self.forward_step(input_data, model, loss_module)
            loss = output['loss']

            if mode == 'Train':
                optimizer.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                optimizer.step()

                if scheduler != None:
                    scheduler.step()

            loss = loss.detach()
            metrics = output['metrics'].detach()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
            if process_id == 0:
                loss /= self.world_size
                metrics /= self.world_size
                metrics_summary = self.metrics_summary(loss.detach(), metrics.detach())
                bar_dict = self._metrics_arrange(mode, metrics_summary)
                if mode == "Train":
                    bar_dict['lr'] = '{:.6f}'.format(optimizer.param_groups[0]['lr'])
                loader.set_postfix(bar_dict)
                metrics_list.append(metrics.detach().cpu().numpy())
        
        if process_id == 0:
            all_result, table = self._epoch_result(epoch, mode, bar_dict, metrics_list)
            logger.info(all_result)
            if table != None:
                logger.info(table)

            self.sort_list[mode].append((epoch, float(bar_dict[self.sort_key])))
        
    def _epoch_result(self, epoch, mode, bar_dict, metrics_list):
        all_info = '{}_Epoch_{}:'.format(mode, epoch) + ','.join(key + '=' + bar_dict[key].strip() for key in bar_dict.keys())
        if mode == 'Val':
            metrics = sum(metrics_list) / len(metrics_list)
            table = PrettyTable()
            table.add_column(' ', self.class_tag)
            for i, k in enumerate(self.metrics_tag):
                table.add_column(k, list(metrics[:, i]))
        else:
            table = None

        return all_info, table

    def _metrics_arrange(self, mode, summary):
        bar_dict = OrderedDict()
        for k, v in summary.items():
            self.recorder[mode][k].append(v.item())
            self.window_dict[k].update(v.item())
            bar_dict[k] = '{:.4f}'.format(self.window_dict[k].value)

        return bar_dict

    def metrics_summary(self, loss, metrics):
        '''Metrics summary
        Arg:
            loss: Tensor
            metrics: Tensor, len(class_tag)*len(metrics_tag)
        Return:
            summary: OrderedDict
        '''
        summary = OrderedDict()
        summary['loss'] = loss.detach().cpu().numpy()
        metrics_mean = metrics.detach().mean(dim=0).cpu().numpy()
        for i, k in enumerate(self.metrics_key[1:]):
            summary[k] = metrics_mean[i]

        return summary

    def _recorder_init(self):
        self.recorder = {
            'Train': OrderedDict(),
            'Val': OrderedDict()
        }

        self.recorder['Train']['loss'] = []
        self.recorder['Val']['loss'] = []

        for k in self.metrics_key:
            self.recorder['Train'][k] = []
            self.recorder['Val'][k] = []

        self.window_dict = OrderedDict()
        for k in self.recorder['Train'].keys():
            self.window_dict[k] = window()

    def optimizer_init(self, model, lr):
        """Optimizer initialization
        Args:
            model: torch.nn
        Return:
            optimizer, scheduler
        """
        optimizer = self.optimizer(model.parameters(), **self.optimizer_config)
        optimizer.param_groups[0]['lr'] = lr
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_config)
        else:
            scheduler = None
        
        return optimizer, scheduler

    def forward_step(self, input_data, model, loss_module):
        """Forward
        Args:
            input: tuple
            model: torch.nn
        Return:
            output: dict: {'loss':Tensor, 'output':Tensor, 'metrics':Tensor}
        """
        x, y = input_data
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        output = model(x)
        loss = loss_module(output, y)
        
        output_result = {'loss': loss, 'output': output, 'metrics': torch.Tensor()}

        return output_result

    def _dataset_init(self):
        # train_dataset = self.dataset(**self.train_data_config)

        # if self.val_data_config != None:
        #     val_dataset = self.dataset(**self.val_data_config)
        # else:
        #     val_dataset = None

        # return train_dataset, val_dataset

        self.train_dataset = self.dataset(**self.train_data_config)

        if self.val_data_config != None:
            self.val_dataset = self.dataset(**self.val_data_config)
        else:
            self.val_dataset = None

    def _logger_init(self, process_id):
        logger = logging.getLogger(self.model_tag + str(process_id))
        logger.setLevel(logging.INFO)
        logger.propagate = False

        ch1 = logging.StreamHandler()
        ch1.setLevel(logging.WARNING)
        ch1.setFormatter(logging.Formatter('%(name)s-Info:%(message)s'))
        logger.addHandler(ch1)

        # ch2 = logging.FileHandler(self.log_path.split('.')[0]+'{}.log'.format(process_id), 'w')
        # ch2.setLevel(logging.DEBUG)
        # ch2.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(ch2)

        ch3 = logging.FileHandler(self.log_path, 'w')
        ch3.setLevel(logging.INFO)
        ch3.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch3)

        return logger

    def _PrintConfig(self):
        config = "Model Tag: {}\nRecord Path: {}\nClass Num: {}\nLog Path: {}\nMetrics Num: {}\nLearning Rate: {}\nEpochs: {}\nEval T: {}\nWarmingUp T: {}\nWorkers: {}\nBatchSize: {}\nUseAmp: {}\n".format(
            self.model_tag, self.record_path, len(self.class_tag), self.log_path, len(self.metrics_key), self.lr, self.epochs, self.eval_T, self.warmingup_T, self.workers, self.batch_size, self.use_amp
        )

        print(config)

class window:
    def __init__(self):
        self.sum = 0
        self.iter_num = 0
    
    def update(self, value):
        if np.isnan(value) == False:
            self.sum += value
            self.iter_num += 1
    
    def reset(self):
        self.sum = 0
        self.iter_num = 0
    
    @property
    def value(self):
        if self.iter_num != 0:
            return self.sum / self.iter_num
        else:
            return 0

def CheckGrads(optimizer):
    max_grad = [x.grad.abs().max() for x in optimizer.param_groups[0]['params']]
    min_grad = [x.grad.abs().min() for x in optimizer.param_groups[0]['params']]
    max_grad = max(max_grad)
    min_grad = min(min_grad)
    print(max_grad, '\n', min_grad)