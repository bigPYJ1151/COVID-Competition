

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
from tools.preprocess import ImageResample, IIRGaussianSmooth
import torch
import torch.nn as nn
from model.attvnet_grouped import VNet
from tools.metrics import SurfaceDistanceCompute, PNMetricsCompute, compute_dice_coefficient

fold_path = ['../data/splits/test.pth']
model_tag = 'ensemble_fine'
data_path = os.path.join('..', 'data', '7.4newdata')
coarse_label_tag = 'ensemble_coarse'
record_path = os.path.join('..', 'record')
model_tag_list = ['GroupedAttVNet_fine_fold0.pth', 'GroupedAttVNet_fine_fold1.pth', 'GroupedAttVNet_fine_fold2.pth',
'GroupedAttVNet_fine_fold3.pth', 'GroupedAttVNet_fine_fold4.pth']
num_processing = torch.cuda.device_count()
batchsize = 1
record_type = 'softlink'
patchsize = (144, 208, 208)
spacing = (0.35,0.35,0.625)
expand_num = (30, 30, 30)

class SegEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_path = kwargs['eval_config']['label_path']
        self.expand_num = kwargs['model_config']['expand_num']

    def data_preprocess(self, data, global_inform):
        image = ImageResample(data['image'], self.spacing)
        image = sitk.GetArrayFromImage(image)

        clip_max = 900
        clip_min = -100

        image = np.clip(image, clip_min, clip_max)

        coarse_label = sitk.ReadImage(os.path.join(self.label_path, global_inform['tag'], 'predict.nii.gz'))
        
        #IIR guassin
        coarse_label_smooth = IIRGaussianSmooth(coarse_label, 10)
    
        coarse_label = sitk.GetArrayFromImage(coarse_label)
        if coarse_label.shape != image.shape:
            raise "Shape mismatch!"
        
        margin_list = self._ROICalcu(coarse_label)
        new_margin_list = []
        for i, margin in enumerate(margin_list):
            lower, upper = margin

            lower = max(0, lower - self.expand_num[i])
            upper = min(coarse_label.shape[i], upper + self.expand_num[i])

            new_margin_list.append((lower, upper))

        mask = np.zeros_like(image)
        mask[
            new_margin_list[0][0]:new_margin_list[0][1],
            new_margin_list[1][0]:new_margin_list[1][1],
            new_margin_list[2][0]:new_margin_list[2][1],
        ] = 1
        mask = mask.astype('bool')

        mean_val = image[mask].mean()
        std_val = image[mask].std()
        min_val = image[mask].min()

        image[(1 - mask).astype('bool')] = min_val

        pad_s = []
        ROI_se = []
        for i, margin in enumerate(new_margin_list):
            lower, upper = margin
            r = self.patchsize[i] - (upper - lower)
            if r > 0:
                lp = int(r / 2)
                rp = r - int(r / 2)
                
                lower -= lp 
                upper += rp

                lp = max(0, -lower)
                rp = max(0, upper - image.shape[i])
                pad_s.append((lp, rp))
                ROI_se.append((lower + lp, upper + lp))
            else:
                pad_s.append((0, 0))
                ROI_se.append((lower, upper))
        
        coarse_label = sitk.GetArrayFromImage(coarse_label_smooth)
        coarse_label /= coarse_label.max()

        image = np.pad(image, tuple(pad_s), mode='constant', constant_values=min_val)
        coarse_label = np.pad(coarse_label, tuple(pad_s), mode='constant', constant_values=0)

        #normalize 1
        image = (image - mean_val) / std_val

        #normalize 3
        # image = (image - (clip_max + clip_min) / 2) / (clip_max - clip_min)

        global_inform['pad_s'] = pad_s
        global_inform['padded_shape'] = image.shape
        global_inform['ROI_inform'] = ROI_se

        image = np.expand_dims(image, 0)
        coarse_label = np.expand_dims(coarse_label, 0)
        image = np.concatenate((image, coarse_label), axis=0)

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
            'record_path': os.path.join(record_path, model_tag, 'data'),
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
            'compute_metric': False,
            'label_path':os.path.join(record_path, coarse_label_tag, 'data')
        },
        'model_config':{
            'model': [SegModel],
            'model_config': [{'inplane':2, 'plane':3}],
            'model_path':[model_path],
            'patchsize': patchsize,
            'spacing': spacing,
            'expand_num': expand_num
        }
    }
    evaluator = SegEvaluator(**config)
    evaluator.eval()