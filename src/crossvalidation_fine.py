
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
from model.resnet import ResNet, Bottleneck, BasicBlock
from model.loss_v2 import GDiceLoss_v2, GDiceLoss, TverskyLoss, CAMConstrain, CAMConstrainDirection
from tools.metrics import DSCCompute, PNMetricsCompute
from tools.Trainer import Trainer

dataset_name = 'stoic2021'
fold_path = os.path.join('..', 'data', dataset_name, 'splits_cls')
data_path = os.path.join('..', 'data', dataset_name, 'data_2')
# data_path = os.path.join('/home/ps', 'all_data_{}'.format(data_spacing))
# label_path = os.path.join('/home/ps', '{}_fine_duc_ds_gatt_2_4_best'.format(dataset_name))
model_tag = '{}_resnet18_newpreprocess'.format(dataset_name)

batchsize = 8
lr = 1e-4
workers = 5
epoch = 50

class LossModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prob, label):
        loss = F.smooth_l1_loss(prob, label, reduction='none')
        loss = loss.mean(dim=1)

        return loss

class ClassifyModelTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_step(self, input_data, model, loss_module):
        _, image, label = input_data
        image = image.cuda(non_blocking=True)
        label = label.to(torch.float32).cuda(non_blocking=True)
    
        output = model(image)
        loss = loss_module(output, label).mean()

        with torch.no_grad():
            predict = (output > 0.5).type_as(label) 
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
        LoopTimes = 1
        targetFold = "fold{}.pth".format(foldIter)
        metricHistory = []
        
        for loopIter in range(LoopTimes):
            print("Start loop, iter={}".format(loopIter))
            for foldi in os.listdir(fold_path):
                if foldi != targetFold:
                    continue

                val_list = []
                with open(os.path.join(fold_path, foldi), 'rb') as f:
                    val_name = pickle.load(f)

                for fname in val_name:
                    val_list.append(os.path.join(data_path, str(fname)))

                train_list = []
                for fold in os.listdir(fold_path):
                    if fold == foldi:
                        continue

                    with open(os.path.join(fold_path, fold), 'rb') as f:
                        train_name = pickle.load(f)

                    for fname in train_name:
                        train_list.append(os.path.join(data_path, str(fname)))

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
                        'train_config':{'data_path':train_list, 'patch_size':(240,240,240), 'expand_num':(0, 0, 0), 'class_num':2, 'train':True},
                        'val_config':{'data_path':val_list, 'patch_size':(240,240,240), 'expand_num':(0, 0, 0), 'class_num':2, 'train':False}
                    },
                    'train_config':{
                        'model':(ResNet, {"block": BasicBlock, "layers": [2, 2, 2, 2], "sample_input_D": 0, "sample_input_H": 0, "sample_input_W": 0, "num_seg_classes": 2}),
                        'loss_module':(LossModel, {}),
                        'optimizer':(Adam, {'lr':lr, 'weight_decay':1e-5}),
                        # 'scheduler':(CosineAnnealingWarmRestarts, {'T_0':1, 'T_mult':17, 'eta_min':lr / 20}),
                    },
                    'record_config':{
                        'tag':current_tag,
                        'class_tag':['COVID', 'Severe'],
                        'metrics_tag':['Accuracy', 'Precision', 'Recall', "F1"],
                        'metrics_key':['loss', "rloss", 'Accuracy', 'Precision', 'Recall', "F1"],
                        'sort_key':'rloss',
                        'record_path':os.path.join('../record', current_tag, 'model'),
                        'log_path':os.path.join('log', current_tag)
                    },
                    'other_config':{
                        'epochs':epoch, 'batch_size':batchsize, 
                        'workers':workers, 'eval_T':4, 'warmingup_T':2, 'amp_train':False,
                        'pretrian_path': os.path.join('..', 'record', 'pretrain', 'resnet_18.pth')
                    }
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
                
                

        