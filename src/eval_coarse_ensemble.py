

import os 
import pickle
import argparse 
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='0,1', type=str, help='Index of GPU used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

from tools.Evaluator import Evaluator
from tools.postprocess import percentConnectComponents
import torch
import torch.nn as nn
from model.attvnet_grouped import VNet
from tools.metrics import SurfaceDistanceCompute, PNMetricsCompute, compute_dice_coefficient

fold_path = ['../data/splits/test.pth']
model_tag = 'ensemble_coarse'
data_path = os.path.join('..', 'data', '7.4newdata_0.35_0.35_0.625')
record_path = os.path.join('..', 'record')
model_tag_list = ['GroupedAttVNet_fold0.pth', 'GroupedAttVNet_fold1.pth', 'GroupedAttVNet_fold2.pth',
'GroupedAttVNet_fold3.pth', 'GroupedAttVNet_fold4.pth']
num_processing = torch.cuda.device_count()
batchsize = 2
record_type = 'softlink'
patchsize = (144, 208, 208)
spacing = (1,1,1)

class SegEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_preprocess(self, data, global_inform):
        output, global_inform = super().data_preprocess(data, global_inform)

        image = output['image']

        clip_max = 900
        clip_min = -100

        image = np.clip(image, clip_min, clip_max)

        #normalize 1
        image = (image - image.mean()) / image.std()

        #normalize 3
        # image = (image - (clip_max + clip_min) / 2) / (clip_max - clip_min)

        output = {'image': image}

        return output, global_inform

    def metric_compute(self, predict, label, global_inform):
        # dsc, hd, asd = SurfaceDistanceCompute(predict.astype('bool'), label.astype('bool'), list(reversed(global_inform['origin_spacing'])))
        dsc = compute_dice_coefficient(predict.astype('bool'), label.astype('bool'))
        predict = torch.from_numpy(predict).unsqueeze(0).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        jaccard, precision, recall, specificity, accuracy = PNMetricsCompute(predict, label)

        # return [dsc, precision.item(), recall.item(), specificity.item(), hd, asd]
        return [dsc, precision.item(), recall.item(), specificity.item()]

    def postprocess(self, predict, global_inform):
        predict = super().postprocess(predict, global_inform)
        predict = self._One_hot(predict)
        predict = percentConnectComponents(predict)
        predict = predict.argmax(axis=0)

        return predict

class SegModel(nn.Module):
    def __init__(self, inplane, plane):
        super().__init__()
        self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.ReLU)
    
    def forward(self, x):
        out = self.vnet(x)

        return out

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    if os.path.exists(os.path.join(record_path, model_tag)) == False:
        os.mkdir(os.path.join(record_path, model_tag))
        os.mkdir(os.path.join(record_path, model_tag, 'data'))
        os.mkdir(os.path.join(record_path, model_tag, 'model'))

    model_path = []
    for tag in model_tag_list:
        pth = os.path.join(record_path, tag, 'model', 'best_model.pth')
        pth = os.path.abspath(pth)
        model_path.append(pth)
        os.symlink(pth, os.path.join(record_path, model_tag, 'model', '{}.pth'.format(tag)))

    record_path = os.path.join(record_path, model_tag, 'data')
    
    tag_list = []
    for fpth in fold_path:
        with open(fpth, 'rb') as f:
            tag_list = pickle.load(f)

    path_list = []
    data_path = os.path.abspath(data_path)
    for tag in tag_list:
        path_list.append(os.path.join(data_path, tag))

    config = {
        'record_config':{
            'tag': model_tag, 
            'record_path': record_path,
            'class_tag':['background', 'PV', 'LA'],
            'metrics_tag':['DSC', 'Precision', 'Recall', 'Specificity'],
            # 'metrics_tag':['DSC', 'Precision', 'Recall', 'Specificity', 'HD', 'ASD'],
            'record_type': record_type,
        },
        'eval_config':{
            'path_list': path_list,
            'tag_list': tag_list,
            'num_processing': num_processing,
            'batchsize': batchsize,
            'compute_metric': False
        },
        'model_config':{
            'model': [SegModel],
            'model_config': [{'inplane':1, 'plane':3}],
            'model_path':[model_path],
            'patchsize': patchsize,
            'spacing': spacing,
        }
    }
    evaluator = SegEvaluator(**config)
    evaluator.eval()