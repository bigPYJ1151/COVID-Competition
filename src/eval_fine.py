

import os 
import pickle
import argparse 
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default='1', type=str, help='Index of GPU used')
parser.add_argument('-f', '--fold', type=str, help="Fold num to evaluate")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 

from tools.Evaluator import Evaluator
import torch
import torch.nn as nn
from model.resnet import ResNet, BasicBlock
from tools.metrics import SurfaceDistanceCompute, PNMetricsCompute, compute_dice_coefficient
from tools.postprocess import percentConnectComponents
from tools.preprocess import ImageResample, ClipandNormalize, IIRGaussianSmooth

dataset_name = 'stoic2021'
fold_path = [os.path.join('..', 'data', dataset_name, 'splits_cls', 'fold{}.pth'.format(args.fold))]
data_path = os.path.join('..', 'data', dataset_name, 'data_3')
model_tag = '{}_resnet18_newpreprocess_v3'.format(dataset_name)
model_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'model', 'best_model.pth')
record_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'data')

num_processing = torch.cuda.device_count()
batchsize = 1	
record_type = 'hardlink'

class ClassifyEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dataload(self, fold_path, global_inform):
        output = {}
        with open(os.path.join(fold_path), 'rb') as f:
            output['image'] = pickle.load(f)
        
        return output, global_inform

    def data_preprocess(self, data, global_inform):
        image = np.expand_dims(data['image'], 0)

        output = {'image': image}

        return output, global_inform

    def metric_compute(self, predict, label, global_inform):
        return None

    def postprocess(self, predict, global_inform):
        prob = torch.sigmoid(predict) 

        return prob.cpu().numpy()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    tag_list = []
    for pth_path in fold_path:
        with open(pth_path, 'rb') as f:
            tag_list += pickle.load(f)

    data_path = os.path.abspath(data_path)
    path_list = []
    for tag in tag_list:
        path_list.append(os.path.join(data_path, str(tag)))

    config = {
        'record_config':{
            'tag': model_tag, 
            'record_path': record_path,
            'class_tag':['COVID', 'Serve'],
            'record_type': record_type,
        },
        'eval_config':{
            'path_list': path_list,
            'tag_list': tag_list,
            'num_processing': num_processing,
            'batchsize': batchsize,
            'compute_metric': False,
        },
        'model_config':{
            'model': [ResNet],
            'model_config': [{"block": BasicBlock, "layers": [2, 2, 2, 2], "sample_input_D": 0, "sample_input_H": 0, "sample_input_W": 0, "num_seg_classes": 2}],
            'model_path':[[model_path]],
            'patchsize': (120,240,240),
        }
    }
    evaluator = ClassifyEvaluator(**config)
    evaluator.eval()