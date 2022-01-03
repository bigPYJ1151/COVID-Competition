
import os 
import argparse 
from collections import OrderedDict
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='2,3', type=str, help='Index of GPU used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data import SegData
from model.attvnet_grouped import VNet
from model.loss_v2 import GDiceLoss_v2
from tools.metrics import DSCCompute, PNMetricsCompute
from tools.Trainer import Trainer

fold_path = '/home/sjtu/data/splits'
model_tag = 'GroupedAttVNet'
batchsize = 1
epoch = 120
lr = 1e-3
workers = 8
pretrain = None
check_point = None

class SegModel(nn.Module):
    def __init__(self, inplane, plane):
        super().__init__()
        self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
    
    def forward(self, x):
        out = self.vnet(x)

        return out

class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = GDiceLoss_v2()

    def forward(self, logit, label):
        prob = torch.softmax(logit, dim=1)
        loss = self.dice(prob, label)

        return loss

class ModelTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_step(self, input_data, model, loss_module):
        _, image, label = input_data
        image = image.cuda(non_blocking=True)
        label = label.to(dtype=torch.float32).cuda(non_blocking=True)

        output = model(image)
        loss = loss_module(F.softmax(output, dim=1), label).mean()
        with torch.no_grad():
            predict = output.argmax(dim=1).detach().cpu().numpy()

            predict = np.eye(len(self.class_tag))[predict]
            predict = np.moveaxis(predict, 4, 1)
            predict = torch.from_numpy(predict).type_as(label)
            
            DSC, VOE, VD = DSCCompute(predict, label)
            Jaccard, Precision, Recall, Specificity, Accuracy = PNMetricsCompute(predict, label)
            metrics = torch.cat([DSC.unsqueeze(1), VOE.unsqueeze(1), VD.unsqueeze(1),
            Precision.unsqueeze(1), Recall.unsqueeze(1)], dim=1)
        return {
            'loss': loss,
            'output': output,
            'metrics': metrics,
        }

    def metrics_summary(self, loss, metrics):
        summary = OrderedDict()
        summary['loss'] = loss.detach().cpu().numpy()
        metrics_mean = metrics.detach().mean(dim=0).cpu().numpy()
        summary['DSCE'] = metrics[1:, 0].mean()
        for i, k in enumerate(self.metrics_key[2:]):
            summary[k] = metrics_mean[i]

        return summary

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 

    with open(os.path.join(fold_path, 'train.pth'), 'rb') as f:
        train_fname = pickle.load(f)

    with open(os.path.join(fold_path, 'val.pth'), 'rb') as f:
        val_fname = pickle.load(f)

    train_list = []
    val_list = []
    for fname in train_fname:
        train_list.append(os.path.join('/hdd5', 'sjtu', 'downsampled_data', fname))

    for fname in val_fname:
        val_list.append(os.path.join('/hdd5', 'sjtu', 'downsampled_data', fname))

    if os.path.exists(os.path.join('../record', model_tag)) == False:
        if os.path.exists(os.path.join('../record', model_tag)) == False:
            os.mkdir(os.path.join('../record', model_tag))
        os.mkdir(os.path.join('../record', model_tag, 'model'))
        os.mkdir(os.path.join('../record', model_tag, 'data'))
    
    if os.path.exists(os.path.join('log', model_tag)) == False:
        os.mkdir(os.path.join('log', model_tag))

    config = {
        'data_config':{
            'dataset':SegData,
            'train_config':{'data_path':train_list, 'patch_size':(144, 208, 208), 'clip_max': 900, 'clip_min':0, 'train':True},
            'val_config':{'data_path':val_list, 'patch_size':(144, 208, 208), 'clip_max': 900, 'clip_min':0, 'train':False}
        },
        'train_config':{
            'model':(SegModel, {'inplane':1, 'plane':2}),
            'loss_module':(LossModel, {}),
            'optimizer':(Adam, {'lr':lr, 'weight_decay':1e-5}),
            'scheduler':(CosineAnnealingWarmRestarts, {'T_0':1, 'T_mult':5, 'eta_min':lr / 20}),
        },
        'record_config':{
            'tag':model_tag,
            'class_tag':['background', 'Co'],
            'metrics_tag':['DSC', 'VOE', 'VD', 'Precision', 'Recall'],
            'metrics_key':['loss', 'DSCE', 'DSC', 'VOE', 'VD', 'Precision', 'Recall'],
            'sort_key':'DSCE',
            'record_path':os.path.join('../record', model_tag, 'model'),
            'log_path':os.path.join('log', model_tag)
        },
        'other_config':{
            'epochs':epoch, 'batch_size':batchsize, 
            'workers':workers, 'eval_T':1, 'warmingup_T':2}
    }

    trainer = ModelTrainer(**config)
    trainer.train()
