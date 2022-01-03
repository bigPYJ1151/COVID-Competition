
import os 
import argparse 
from collections import OrderedDict
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='1', type=str, help='Index of GPU used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data import ClassifyDataExtraC
from model.classifer import VNet
# from model.unet import UNet3D
from model.loss_v2 import GDiceLoss_v2, GDiceLoss, TverskyLoss, CAMConstrain, CAMConstrainDirection
from tools.metrics import DSCCompute, PNMetricsCompute
from tools.Trainer import Trainer

dataset_name = 'TAPVC'
data_spacing = '0.35_0.35_0.625'

fold_path = os.path.join('..', 'data', dataset_name, 'splits_cls')
# data_path = os.path.join('..', 'data', dataset_name, 'all_data_{}'.format(data_spacing))
# label_path = os.path.join('..', 'data', "TAPVC_fine", '{}_fine_duc_ds_gatt_2_4_best'.format(dataset_name))
data_path = os.path.join('/home/ps', 'all_data_{}'.format(data_spacing))
label_path = os.path.join('/home/ps', '{}_fine_duc_ds_gatt_2_4_best'.format(dataset_name))
model_tag = '{}_classify_softmax_resin_cam0.1_v3'.format(dataset_name)

batchsize = 16
lr = 1e-4
workers = 4
epoch = 100

class ClassifyModel(nn.Module):
    def __init__(self, inplane, plane):
        super().__init__()
        # self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
        # self.unet = UNet3D(inplane, plane, layer_order='cri')
        self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.LeakyReLU, n_filters=8)

    def forward(self, x):
        out = self.vnet(x)
        # out = self.unet(x)

        return out

class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.camC = CAMConstrain(weight=0.1)
        self.camD = CAMConstrainDirection(weight=0.1)
        # self.ce = nn.CrossEntropyLoss(torch.Tensor([1,3]), reduction='none')

    def forward(self, logit, label, cam, camLabel):
        # loss = self.ce(logit, label)
        ce = self.ce(logit, label)
        camC = self.camC(cam, camLabel)
        camD = self.camD(cam, camLabel, label)
        loss = ce + camC + camD
        
        return loss

class ClassifyModelTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_step(self, input_data, model, loss_module):
        _, image, label, imageLabel, features = input_data
        image = image.cuda(non_blocking=True)
        label = label.to(torch.long).cuda(non_blocking=True)
        imageLabel = imageLabel.to(torch.long).cuda(non_blocking=True)
    
        output, cam = model(image)
        loss = loss_module(output, label, cam, imageLabel).mean()

        with torch.no_grad():
            label = F.one_hot(label)

            predict = torch.argmax(output.detach(), dim=1)
            predict = F.one_hot(predict)

            Jaccard, Precision, Recall, Specificity, Accuracy, F1 = PNMetricsCompute(predict, label)
            metrics = torch.cat([Accuracy.unsqueeze(1), Precision.unsqueeze(1), Recall.unsqueeze(1), F1.unsqueeze(1)], dim=1)
        return {
            'loss': loss,
            'output': output,
            'metrics': metrics,
        }

    def metrics_summary(self, loss, metrics):
        summary = OrderedDict()
        summary['loss'] = loss.detach().cpu().numpy()
        summary['rloss'] = 1 - summary['loss']
        metrics_mean = metrics.detach().mean(dim=0).cpu().numpy()
        for i, k in enumerate(self.metrics_key[2:]):
            summary[k] = metrics_mean[i]

        return summary

def loadBestMetric(recordPath):
    if "best_model.pth" in os.listdir(recordPath):
        data = torch.load(os.path.join(recordPath, "best_model.pth"))
        return data['record']['Val']['rloss'].pop()
    else:
        return None

def backupBestModel(recordPath):
    if "best_model.pth" in os.listdir(recordPath): 
        if "prev_best_model.pth" in os.listdir(recordPath):  
            os.remove(os.path.join(recordPath, "prev_best_model.pth"))

        os.rename(os.path.join(recordPath, "best_model.pth"), os.path.join(recordPath, "prev_best_model.pth"))

def recoverBackupModel(recordPath):
    if "prev_best_model.pth" in os.listdir(recordPath): 
        if "best_model.pth" in os.listdir(recordPath):  
            os.remove(os.path.join(recordPath, "best_model.pth"))

        os.rename(os.path.join(recordPath, "prev_best_model.pth"), os.path.join(recordPath, "best_model.pth"))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    for foldIter in range(5): 
        PrevBestMetric = None
        LoopTimes = 10
        targetFold = "fold{}.pth".format(foldIter)
        metricHistory = []
        
        for loopIter in range(LoopTimes):
            print("Start loop, iter={}".format(loopIter))
            for foldi in os.listdir(fold_path):
                if foldi != targetFold:
                    continue

                val_list = []
                val_label_list = []
                with open(os.path.join(fold_path, foldi), 'rb') as f:
                    val_name = pickle.load(f)

                for fname in val_name:
                    val_list.append(os.path.join(data_path, fname))
                    val_label_list.append(os.path.join(label_path, fname))

                train_list = []
                train_label_list = []
                for fold in os.listdir(fold_path):
                    if fold == foldi:
                        continue

                    with open(os.path.join(fold_path, fold), 'rb') as f:
                        train_name = pickle.load(f)

                    for fname in train_name:
                        train_list.append(os.path.join(data_path, fname))
                        train_label_list.append(os.path.join(label_path, fname))

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

                PrevBestMetric = loadBestMetric(os.path.join('../record', current_tag, 'model'))
                backupBestModel(os.path.join('../record', current_tag, 'model'))

                config = {
                    'data_config':{
                        'dataset':ClassifyDataExtraC,
                        'train_config':{'data_path':train_list, 'label_path':train_label_list, 'patch_size':(128, 160, 208), 'expand_num':(0, 0, 0), 'clip_max': 1200, 'clip_min':-100, 'class_num':2, 'train':True},
                        'val_config':{'data_path':val_list, 'label_path':val_label_list, 'patch_size':(128, 160, 208), 'expand_num':(0, 0, 0), 'clip_max': 1200, 'clip_min':-100, 'class_num':2, 'train':False}
                    },
                    'train_config':{
                        'model':(ClassifyModel, {'inplane':1, 'plane':2}),
                        'loss_module':(LossModel, {}),
                        'optimizer':(Adam, {'lr':lr, 'weight_decay':1e-5}),
                        # 'scheduler':(CosineAnnealingWarmRestarts, {'T_0':1, 'T_mult':17, 'eta_min':lr / 20}),
                    },
                    'record_config':{
                        'tag':current_tag,
                        'class_tag':['non-PVO', 'PVO'],
                        'metrics_tag':['Accuracy', 'Precision', 'Recall', "F1"],
                        'metrics_key':['loss', "rloss", 'Accuracy', 'Precision', 'Recall', "F1"],
                        'sort_key':'rloss',
                        'record_path':os.path.join('../record', current_tag, 'model'),
                        'log_path':os.path.join('log', current_tag)
                    },
                    'other_config':{
                        'epochs':epoch, 'batch_size':batchsize, 
                        'workers':workers, 'eval_T':4, 'warmingup_T':2, 'amp_train':False}
                }
                trainer = ClassifyModelTrainer(**config)
                try:
                    trainer.train()
                except torch.multiprocessing.ProcessExitedException:
                    pass

                currentMetric = loadBestMetric(os.path.join('../record', current_tag, 'model'))
                if PrevBestMetric == None or currentMetric > PrevBestMetric:
                    PrevBestMetric = currentMetric
                    metricHistory.append(PrevBestMetric)
                else:
                    recoverBackupModel(os.path.join('../record', current_tag, 'model'))

        print("Loop training report:\nfold:{}\ntimes:{}\nrecord:{}\n".format(
            targetFold, LoopTimes, metricHistory
        ))    
                
                

        