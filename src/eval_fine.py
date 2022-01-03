

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
from model.classifer import VNet
# from model.unet import UNet3D
from tools.metrics import SurfaceDistanceCompute, PNMetricsCompute, compute_dice_coefficient
from tools.postprocess import percentConnectComponents
from tools.preprocess import ImageResample, ClipandNormalize, IIRGaussianSmooth

dataset_name = 'TAPVC'
data_spacing = '0.35_0.35_0.625'

fold_path = [os.path.join('..', 'data', dataset_name, 'splits_cls', 'fold{}.pth'.format(args.fold))]
data_path = os.path.join('..', 'data', dataset_name, 'all_data_{}'.format(data_spacing))
model_tag = '{}_classify_softmax_resin_cam0.1_v3'.format(dataset_name)
model_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'model', 'best_model.pth')
record_path = os.path.join('..', 'record', '{}_fold{}.pth'.format(model_tag, args.fold), 'data')
coarse_label_tag = os.path.join('..', 'data', "TAPVC_fine", '{}_fine_duc_ds_gatt_2_4_best'.format(dataset_name))

num_processing = torch.cuda.device_count()
batchsize = 1	
record_type = 'hardlink'
patchsize = (128, 160, 208)
spacing = (0.35, 0.35, 0.625)
expand_num = (0, 0, 0)

class ClassifyEvaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_path = kwargs['eval_config']['label_path']
        self.expand_num = kwargs['model_config']['expand_num']

    def data_preprocess(self, data, global_inform):
        image = ImageResample(data['image'], self.spacing)
        image = sitk.GetArrayFromImage(image)

        clip_max = 1200
        clip_min = -100

        coarse_label = sitk.ReadImage(os.path.join(self.label_path, global_inform['tag'], 'predict.nii.gz'))
        
        #IIR guassin
        # coarse_label_smooth = IIRGaussianSmooth(coarse_label, 10)
    
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
        preImage = image.copy()
        image[(1 - mask).astype('bool')] = clip_min

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
        
        # For IIR smooth 
        # coarse_label = sitk.GetArrayFromImage(coarse_label_smooth)
        # coarse_label /= coarse_label.max()

        image = np.pad(image, tuple(pad_s), mode='constant', constant_values=clip_min)
        preImage = np.pad(preImage, tuple(pad_s), mode='constant', constant_values=clip_min)
        coarse_label = np.pad(coarse_label, tuple(pad_s), mode='constant', constant_values=0)

        global_inform["cropped_image"] = preImage[
            ROI_se[0][0]:ROI_se[0][1],
            ROI_se[1][0]:ROI_se[1][1],
            ROI_se[2][0]:ROI_se[2][1],
        ]

        global_inform["cropped_label"] = coarse_label[
            ROI_se[0][0]:ROI_se[0][1],
            ROI_se[1][0]:ROI_se[1][1],
            ROI_se[2][0]:ROI_se[2][1]
        ]

        #clip and normalize
        image = ClipandNormalize(image, clip_min, clip_max) 
        
        global_inform['pad_s'] = pad_s
        global_inform['padded_shape'] = image.shape
        global_inform['ROI_inform'] = ROI_se

        image = np.expand_dims(image, 0)

        output = {'image': image}

        return output, global_inform

    def metric_compute(self, predict, label, global_inform):
        dsc, hd, asd = SurfaceDistanceCompute(predict.astype('bool'), label.astype('bool'), list(reversed(global_inform['origin_spacing'])))
        dsc = compute_dice_coefficient(predict.astype('bool'), label.astype('bool'))
        predict = torch.from_numpy(predict).unsqueeze(0).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)

        return None
        # return [dsc, precision.item(), recall.item(), specificity.item()]

    def postprocess(self, predict, global_inform):
        prob, cam = predict

        return prob.cpu().numpy(), cam.cpu().numpy()

class ClassifyModel(nn.Module):
    def __init__(self, inplane, plane):
        super().__init__()

        self.vnet = VNet(n_channels=inplane, n_classes=plane, normalization='instancenorm', has_dropout=False, activation=nn.LeakyReLU, n_filters=8)
    
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
            'class_tag':['non-PVO', 'PVO'],
            # 'metrics_tag':['DSC', 'Precision', 'Recall', 'Specificity'],
            'metrics_tag':['DSC', 'Precision', 'Recall', 'Specificity', 'HD', 'ASD'],
            'record_type': record_type,
            'probmap':True,
        },
        'eval_config':{
            'path_list': path_list,
            'tag_list': tag_list,
            'num_processing': num_processing,
            'batchsize': batchsize,
            'compute_metric': False,
            'label_path': coarse_label_tag
        },
        'model_config':{
            'model': [ClassifyModel],
            'model_config': [{'inplane':1, 'plane':2}],
            'model_path':[[model_path]],
            'patchsize': patchsize,
            'spacing': spacing,
            'expand_num':expand_num
        }
    }
    evaluator = ClassifyEvaluator(**config)
    evaluator.eval()