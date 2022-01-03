

import os 
import pickle
import argparse 
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='0,1', type=str, help='Index of GPU used')
parser.add_argument('-f', '--fold', type=str, help="Fold num to evaluate")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

from tools.Evaluator import Evaluator
import torch
import torch.nn as nn
# from model.attvnet_grouped import VNet
# from model.unet import UNet3D
from model.vnet import VNet
from tools.preprocess import ClipandNormalize
from tools.metrics import SurfaceDistanceCompute, PNMetricsCompute, compute_dice_coefficient
from tools.postprocess import percentConnectComponents

dataset_name = 'TAPVC'
data_spacing = '0.35_0.35_0.625'

fold_path = [os.path.join('..', 'data', dataset_name, 'splits', 'fold{}.pth'.format(args.fold))]
data_path = os.path.join('..', 'data', dataset_name, 'all_data_{}'.format(data_spacing))
model_tag = '{}_coarse'.format(dataset_name)
model_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'model', 'best_model.pth')
record_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'data')

num_processing = torch.cuda.device_count()
batchsize = 1
record_type = 'hardlink'
patchsize = (160, 256, 256)
spacing = (1, 1, 1)

class SegEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_preprocess(self, data, global_inform):
        output, global_inform = super().data_preprocess(data, global_inform)

        image = output['image']

        clip_max = 1200
        clip_min = -100

        #clip and normalize
        image = ClipandNormalize(image, clip_min, clip_max)

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
        predict = predict.cpu().numpy()

        #for probmap save
        global_inform['prob_map'] = predict.copy()

        predict = predict.argmax(axis=0)
        predict = self._One_hot(predict)
        predict = percentConnectComponents(predict, percent=0.05)
        predict = predict.argmax(axis=0)

        return predict

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

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    tag_list = []
    for pth_path in fold_path:
        with open(pth_path, 'rb') as f:
            tag_list += pickle.load(f)

    data_path = os.path.abspath(data_path)
    path_list = []
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
            'probmap':False
        },
        'eval_config':{
            'path_list': path_list,
            'tag_list': tag_list,
            'num_processing': num_processing,
            'batchsize': batchsize,
            'compute_metric': True,
            'memory_opt_level': 2,
        },
        'model_config':{
            'model': [SegModel],
            'model_config': [{'inplane':1, 'plane':3}],
            'model_path':[[model_path]],
            'patchsize': patchsize,
            'spacing': spacing,
        }
    }
    evaluator = SegEvaluator(**config)
    evaluator.eval()