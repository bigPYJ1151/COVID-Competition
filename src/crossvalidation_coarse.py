
import os 
import argparse 
from collections import OrderedDict
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='0', type=str, help='Index of GPU used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data import SegData
# from model.attvnet_grouped import VNet
from model.vnet import VNet
# from model.unet import UNet3D
from model.loss_v2 import GDiceLoss_v2, GDiceLoss, TverskyLoss, TverskyGDiceLoss
from tools.metrics import DSCCompute, PNMetricsCompute
from tools.Trainer import Trainer

dataset_name = 'TAPVC'
data_spacing = '1_1_1'

fold_path = os.path.join('..', 'data', dataset_name, 'splits')
data_path = os.path.join('..', 'data', dataset_name, 'all_data_{}'.format(data_spacing))
model_tag = '{}_coarse_{}'.format(dataset_name, "tversky_0.3_0.7")
batchsize = 2
lr = 1e-3
workers = 2
epoch = 100

class SegModel(nn.Module):
    def __init__(self, inplane, plane):
        super().__init__()
        # self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
        self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False) 
        # self.unet = UNet3D(inplane, plane, layer_order='cri')

    def forward(self, x):
        out = self.vnet(x)
        # out = self.unet(x)

        return out

class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.dice = GDiceLoss()
        self.tver = TverskyLoss(0.3, 0.7)

    def forward(self, logit, label):
        prob = torch.softmax(logit, dim=1)
        loss = self.tver(prob, label)
        # loss = self.dice(prob, label)

        return loss

class SegModelTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_step(self, input_data, model, loss_module):
        _, image, label = input_data
        image = image.cuda(non_blocking=True)
        label = label.to(dtype=torch.float32).cuda(non_blocking=True)
    
        output = model(image)
        loss = loss_module(output, label).mean()
        with torch.no_grad():
            predict = output.argmax(dim=1).detach()
            predict = F.one_hot(predict, len(self.class_tag)).movedim(4, 1)

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

    for foldi in os.listdir(fold_path):

        val_list = []
        with open(os.path.join(fold_path, foldi), 'rb') as f:
            val_name = pickle.load(f)

        for fname in val_name:
            val_list.append(os.path.join(data_path, fname))

        train_list = []
        for fold in os.listdir(fold_path):
            if fold == foldi:
                continue

            with open(os.path.join(fold_path, fold), 'rb') as f:
                train_name = pickle.load(f)

            for fname in train_name:
                train_list.append(os.path.join(data_path, fname))

        for p in train_list:
            if p in val_list:
                raise Exception('{} duplicated!!!'.format(p))
        
        current_tag = '{}_{}'.format(model_tag, foldi)

        if os.path.exists(os.path.join('../record', current_tag)) == False:
            if os.path.exists(os.path.join('../record', current_tag)) == False:
                os.mkdir(os.path.join('../record', current_tag))
            os.mkdir(os.path.join('../record', current_tag, 'model'))
            os.mkdir(os.path.join('../record', current_tag, 'data'))
        
        if os.path.exists(os.path.join('log', current_tag)) == False:
            os.mkdir(os.path.join('log', current_tag))

        config = {
            'data_config':{
                'dataset':SegData,
                'train_config':{'data_path':train_list, 'patch_size':(160, 256, 256), 'clip_max': 1200, 'clip_min':-100, 'class_num':3, 'train':True},
                'val_config':{'data_path':val_list, 'patch_size':(160, 256, 256), 'clip_max': 1200, 'clip_min':-100, 'class_num':3, 'train':False}
            },
            'train_config':{
                'model':(SegModel, {'inplane':1, 'plane':3}),
                'loss_module':(LossModel, {}),
                'optimizer':(Adam, {'lr':lr, 'weight_decay':1e-5}),
                'scheduler':(CosineAnnealingWarmRestarts, {'T_0':1, 'T_mult':14, 'eta_min':lr / 20}),
            },
            'record_config':{
                'tag':current_tag,
                'class_tag':['background', 'PV', 'LA'],
                'metrics_tag':['DSC', 'VOE', 'VD', 'Precision', 'Recall'],
                'metrics_key':['loss', 'DSCE', 'DSC', 'VOE', 'VD', 'Precision', 'Recall'],
                'sort_key':'DSCE',
                'record_path':os.path.join('../record', current_tag, 'model'),
                'log_path':os.path.join('log', current_tag)
            },
            'other_config':{
                'epochs':epoch, 'batch_size':batchsize, 
                'workers':workers, 'eval_T':1, 'warmingup_T':2, 'amp_train':False}
        }
        trainer = SegModelTrainer(**config)
        trainer.train()
                
                

        