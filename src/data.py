
import os
import pickle 
import torch 
from torch.utils.data import Dataset
import SimpleITK as sitk 
import numpy as np 
from tools import augmentation as aug 
from tools.preprocess import IIRGaussianSmooth, ClipandNormalize

from tqdm import tqdm 

class SegData(Dataset):

    def __init__(self, data_path, patch_size, clip_max, clip_min, class_num, train=True):
        super().__init__()

        self.data_path = data_path
        self.patch_size = patch_size
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.class_num = class_num
        self.train = train

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        clip_max = self.clip_max
        clip_min = self.clip_min

        path = self.data_path[index]

        image = sitk.ReadImage(os.path.join(path, 'im.nii.gz'))
        label = sitk.ReadImage(os.path.join(path, 'mask.nii.gz'))

        # if self.train:
        #     transforms = []
            # if np.random.uniform() > 0.5:
            #     transforms.append(aug.Rotation3D(
            #         image, 
            #         angle=(20 * np.random.rand() - 10, 20 * np.random.rand() - 10, 20 * np.random.rand() - 10),
            #         center = tuple(np.array(image.GetSize()) / 2)
            #     ))

            # if np.random.uniform() > 0.5:
            #     scale = 0.8 + 0.4 * np.random.rand()
            #     transforms.append(aug.Scale3D(
            #         image,
            #         scale=(scale, scale, scale),
            #         center = tuple(np.array(image.GetSize()) / 2),
            #     ))

            # transforms = aug.ComposeTransforms(transforms)
            # image = aug.ExecuteTransform(image, transforms, False, clip_min)
            # label = aug.ExecuteTransform(label, transforms, True)

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        image, label, pad_s = self._Padding(image, label, clip_min)

        #clip and normalize
        image = ClipandNormalize(image, clip_min, clip_max)
        
        z, y, x = image.shape
        zl, yl, xl = self.patch_size
        zs, ys, xs = np.random.randint(0, z - zl + 1), np.random.randint(0, y - yl + 1), np.random.randint(0, x - xl + 1)
        # zs, ys, xs = (z - zl + 1) // 2, (y - yl + 1) // 2, (x - xl + 1) // 2

        image = image[zs:zs+zl, ys:ys+yl, xs:xs+xl]
        label = label[zs:zs+zl, ys:ys+yl, xs:xs+xl]

        if self.train:
            if np.random.uniform() > 0.5:
                image = image * (0.9 + 0.2 * np.random.rand()) + (0.2 * np.random.rand() - 0.1)

            if np.random.uniform() > 0.5:
                sigma =  np.random.uniform() / 100
                image = self._randomNoise(image, sigma)

        onehot_label = self._OneHot(label)
        image = np.expand_dims(image, axis=0)

        return path, image.astype('float32'), onehot_label.astype('int64')

        # return path, image, label

    def _OneHot(self, label):
        onehot_label = np.eye(self.class_num)[label]
        onehot_label = np.moveaxis(onehot_label, 3, 0)

        return onehot_label

    def _Padding(self,image, label, pad_val):
        pad_s = []
        for i, v in enumerate(image.shape):
            r = self.patch_size[i] - v
            if r > 0:
                pad_s.append((int(r / 2), r - int(r / 2)))
            else:
                pad_s.append((0, 0))

        image = np.pad(image, tuple(pad_s), mode='constant', constant_values=pad_val)
        label = np.pad(label, tuple(pad_s), mode='constant', constant_values=0)

        return image, label, pad_s

    def _randomNoise(self, image, sigma):
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * sigma, 2 * sigma)
        image = image + noise

        return image

class ClassifyDataExtraC(SegData):

    def __init__(self, data_path, label_path, patch_size, expand_num, clip_max, clip_min, class_num, train=True):
        super().__init__(data_path, patch_size, clip_max, clip_min, class_num, train)

        self.label_path = label_path
        self.expand_num = expand_num
        self.tapvcInfo = {}
        with open(os.path.join("..", "data", "tapvc_info.pth"), "rb") as f:
            self.tapvcInfo = pickle.load(f)

    def __getitem__(self, index):
        # patchsize (z,y,x) (128,160,208)
        clip_max = self.clip_max
        clip_min = self.clip_min

        path = self.data_path[index]
        coarse_label_path = self.label_path[index]

        image = sitk.ReadImage(os.path.join(path, 'im.nii.gz'))
        coarse_label = sitk.ReadImage(os.path.join(coarse_label_path, 'predict.nii.gz'))
        label = sitk.ReadImage(os.path.join(path, 'mask.nii.gz'))

        image, label, coarse_label = self._ROIextract(image, label, coarse_label)

        if self.train:
            transforms = []
            if np.random.uniform() > 0.5:
                transforms.append(aug.Rotation3D(
                    image, 
                    angle=(20 * np.random.rand() - 10, 20 * np.random.rand() - 10, 20 * np.random.rand() - 10),
                    center = tuple(np.array(image.GetSize()) / 2)
                ))

            # if np.random.uniform() > 0.5:
            #     scale = 0.8 + 0.4 * np.random.rand()
            #     transforms.append(aug.Scale3D(
            #         image,
            #         scale=(scale, scale, scale),
            #         center = tuple(np.array(image.GetSize()) / 2),
            #     ))

            transforms = aug.ComposeTransforms(transforms)
            image = aug.ExecuteTransform(image, transforms, False, clip_min)
            label = aug.ExecuteTransform(label, transforms, True)
            coarse_label = aug.ExecuteTransform(coarse_label, transforms, True)

        #IIR Guassin
        # coarse_label = IIRGaussianSmooth(coarse_label, 10)

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        coarse_label = sitk.GetArrayFromImage(coarse_label)

        image, label, pad_s = self._Padding(image, label, clip_min)
        coarse_label = np.pad(coarse_label, tuple(pad_s), mode='constant', constant_values=0)

        patientName = path.split('/')[-1]
        tapvcItem = self.tapvcInfo[patientName]
        label2 = tapvcItem["label2"]
        feature = np.array(tapvcItem["feature"])

        #For IIR
        # coarse_label /= coarse_label.max()

        #clip and normalize
        image = ClipandNormalize(image, clip_min, clip_max)

        if self.train:
            if np.random.uniform() > 0.5:
                image = image * (0.9 + 0.2 * np.random.rand()) + (0.2 * np.random.rand() - 0.1)

            if np.random.uniform() > 0.5:
                sigma =  np.random.uniform() / 100
                image = self._randomNoise(image, sigma)

        image = np.expand_dims(image, axis=0)

        return path, image.astype('float32'), label2, coarse_label.astype('int64'), feature.astype("float32")

    def _ROIextract(self, image, label, coarse_label):
        spacing = image.GetSpacing()
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        coarse_label = sitk.GetArrayFromImage(coarse_label)

        margin_list = self._ROICalcu(coarse_label)
        new_margin_list = []
        for i, margin in enumerate(margin_list):
            lower, upper = margin

            lower = max(0, lower - self.expand_num[i])
            upper = min(label.shape[i], upper + self.expand_num[i])

            new_margin_list.append((lower, upper))

        image = image[
            new_margin_list[0][0]:new_margin_list[0][1],
            new_margin_list[1][0]:new_margin_list[1][1],
            new_margin_list[2][0]:new_margin_list[2][1],
        ]
        label = label[
            new_margin_list[0][0]:new_margin_list[0][1],
            new_margin_list[1][0]:new_margin_list[1][1],
            new_margin_list[2][0]:new_margin_list[2][1],
        ]
        coarse_label = coarse_label[
            new_margin_list[0][0]:new_margin_list[0][1],
            new_margin_list[1][0]:new_margin_list[1][1],
            new_margin_list[2][0]:new_margin_list[2][1],
        ]

        image = sitk.GetImageFromArray(image)
        label = sitk.GetImageFromArray(label)
        coarse_label = sitk.GetImageFromArray(coarse_label)
        image.SetSpacing(spacing)
        label.SetSpacing(spacing)
        coarse_label.SetSpacing(spacing)

        return image, label, coarse_label
    
    @staticmethod
    def _ROICalcu(label):
        def findMargin(sum_list):
            for i, v in enumerate(sum_list):
                lower = i
                if v != 0:
                    break

            sum_list.reverse()
            for i, v in enumerate(sum_list):
                upper = len(sum_list) - i
                if v != 0:
                    break
                    
            if upper < lower:
                return upper, lower
            else:
                return lower, upper

        label = label.astype('bool')
        margin_list = []
        for i in range(label.ndim):
            edge_view = np.swapaxes(label, 0, i)
            l = edge_view.shape[0]
            edge_view = edge_view.reshape((l, -1)).sum(axis=1)
            lower, upper = findMargin(list(edge_view))

            margin_list.append((lower, upper))

        return margin_list
        

# if __name__ == "__main__":
#     data = SegData(['../data/1001'], (256, 256, 256), 2200, 800, 8)

#     print(len(data))

#     _, image, label = data[0]

#     image = image.astype('float')
#     label = label.astype('uint8')

#     print(image.shape, label.shape)

#     image = np.moveaxis(image, 0, 3)
#     label = np.moveaxis(label, 0, 3)

#     image = sitk.GetImageFromArray(image)
#     label = sitk.GetImageFromArray(label)

#     sitk.WriteImage(image, os.path.join('../1001_pro/im.nii.gz'))
#     sitk.WriteImage(label, os.path.join('../1001_pro/mask.nii.gz'))