
import os
import abc
import logging
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from multiprocessing import Pool, queues, Manager
from tqdm import tqdm
from collections import OrderedDict
import SimpleITK as sitk
from tools.preprocess import ImageResample 
import pandas as pds
import time
import copy
import pickle


class Evaluator:
    '''
    Overwrite:
    Must:
        data_preprocess()
    Optional:
        forward_step()
        dataload()
        patch_extract()
        generate_probmap()
        label_reconstruction()
        postprocess()
        metric_compute()
    '''

    def __init__(self, **kwargs):
        '''Basic Evaluator
        Args:
            record_config={
                'tag':str, 
                'record_path':str, 
                'class_tag':list, 
                'metrics_tag':list, default:[]
                'record_type':str, None|softlink|hardlink, default:None
            }
            eval_config={
                'path_list': list, 
                'tag_list': list,
                'num_processing': int, default:Number of GPUs
                'batchsize': int, default: 1
                'compute_metric': bool, default:True
                'target_spacing': list, (xs,ys,zs), default:None
                'memory_opt_level': 0-3, default:0
                                    0: reconstructed maps and patch predict maps are stored on GPU,
                                    1: patch predict maps are stored on GPU,
                                    2: reconstructed maps maps are stored on GPU,
                                    3: All are stored on CPU
            }
            model_config={
                'model': list of class of model, [model1, model2]
                'model_config': list of dict, [{config1}, {config2}]
                'model_path': list of str list,[['path1', 'path2'], ['path1']]
                'patchsize': tuple, (z,y,x)
                'spacing': tuple, (x, y, z)
            }
        '''

        self.model_tag = kwargs['record_config']['tag']
        self.record_path = kwargs['record_config']['record_path']
        self.class_tag = kwargs['record_config']['class_tag']
        self.metrics_tag = kwargs['record_config'].get('metrics_tag', [])
        self.image_record_type = kwargs['record_config'].get('record_type', 'None')
        self.prob_map = kwargs['record_config'].get('probmap', False) 

        if self.image_record_type == 'None':
            self.image_record_type = 0
        elif self.image_record_type == 'softlink':
            self.image_record_type = 1
        elif self.image_record_type == 'hardlink':
            self.image_record_type = 2
        else:
            raise Exception('Image record type wrong!')

        self.num_gpu = torch.cuda.device_count()

        self.data_path_list = kwargs['eval_config']['path_list']
        self.data_tag_list = kwargs['eval_config']['tag_list']
        self.num_processing = kwargs['eval_config'].get('num_processing', self.num_gpu)
        self.batchsize = kwargs['eval_config'].get('batchsize', 1)
        self.compute_metric = kwargs['eval_config'].get('compute_metric', True)
        self.memory_opt_level = kwargs['eval_config'].get('memory_opt_level', 0)
        self.target_spacing = kwargs['eval_config'].get('target_spacing', None)
        
        if self.num_processing % self.num_gpu != 0:
            raise Exception('num_processing and num_gpu do not match.')
        else:
            self.num_processing_per_gpu = self.num_processing // self.num_gpu
        
        self.model = kwargs['model_config']['model']
        self.model_config = kwargs['model_config']['model_config']
        self.model_path = kwargs['model_config']['model_path']
        self.patchsize = kwargs['model_config']['patchsize']
        self.spacing = kwargs['model_config']['spacing']

        self.recorder = self._recorder_init()

    def eval(self):
        process_pool = Pool(self.num_processing)
        q = Manager().Queue(len(self.data_tag_list))
        
        results = []
        for i in range(self.num_processing):
            result = process_pool.apply_async(self._eval_procedure, args=(i, q,))
            results.append(result)

        result = results[0].get()
        process_pool.close()
        process_pool.join()

        if self.compute_metric:
            for result in results:
                result = result.get()
                for k, v in result.items():
                    self.recorder[k] += v

        dataframe = pds.DataFrame(self.recorder)
        dataframe.to_csv(os.path.join(self.record_path, 'summary.csv'))

    def _eval_procedure(self, processing_id, message_q):
        gpu_num = processing_id // self.num_processing_per_gpu
        torch.cuda.set_device(gpu_num)

        device = torch.device('cuda', gpu_num)
        model = [self.model[i](**self.model_config[i]).to(device='cpu') for i in range(len(self.model))]

        logger = self._logger_init(processing_id)
        logger.warning('Process:{} launched.'.format(processing_id))

        recorder = self._recorder_init()

        if processing_id == 0:
            bar = tqdm(total=len(self.data_tag_list))
            for index in range(len(self.data_tag_list)):
                message_q.put(index)
        else:
            time.sleep(1)
        
        with torch.no_grad():
            while not message_q.empty():

                if processing_id == 0:
                    update_n = len(self.data_tag_list) - message_q.qsize() - bar.n
                    bar.update(update_n)
        
                try:
                    index = message_q.get(timeout=1)
                except queues.Empty:
                    if processing_id == 0:
                        continue
                    else:
                        break

                data_path = self.data_path_list[index]
                data_tag = self.data_tag_list[index]

                global_inform = {}
                global_inform['tag'] = data_tag
                global_inform['device'] = device

                origin_output, global_inform = self.dataload(data_path, global_inform)
                preprocessed_output, global_inform = self.data_preprocess(origin_output, global_inform)
                global_inform = self.patch_extract(preprocessed_output, global_inform)
                output = self.generate_probmap(preprocessed_output, model, global_inform) 
                output = self.postprocess(output, global_inform)

                if self.compute_metric:
                    label = origin_output['label']
                    label = sitk.GetArrayFromImage(label)
                    recorder = self._metric_compute(output, label, recorder, global_inform)

                store_path = os.path.join(self.record_path, global_inform['tag'])
                if os.path.exists(store_path) == False:
                    os.mkdir(store_path)

                prob, probmap_array = output
                with open(os.path.join(store_path, "prob.pth"), 'wb') as f:
                    pickle.dump(prob, f)

                ROI_se = global_inform["ROI_inform"]

                probmap_array = probmap_array[
                    :,
                    ROI_se[0][0]:ROI_se[0][1],
                    ROI_se[1][0]:ROI_se[1][1],
                    ROI_se[2][0]:ROI_se[2][1]
                ]
                probmap_array = np.moveaxis(probmap_array, 0, 3)
                probmap_array = sitk.GetImageFromArray(probmap_array)
                probmap_array.SetSpacing(global_inform['origin_spacing'])
                probmap_array.SetOrigin(global_inform['origin_origin'])
                probmap_array.SetDirection(global_inform['origin_direction'])

                sitk.WriteImage(probmap_array, os.path.join(store_path, "cam.nii.gz"))
                
                probmap_array = global_inform["cropped_image"]
                probmap_array = sitk.GetImageFromArray(probmap_array)
                probmap_array.SetSpacing(global_inform['origin_spacing'])
                probmap_array.SetOrigin(global_inform['origin_origin'])
                probmap_array.SetDirection(global_inform['origin_direction'])

                sitk.WriteImage(probmap_array, os.path.join(store_path, "cim.nii.gz"))
                
                probmap_array = global_inform["cropped_label"]
                probmap_array = sitk.GetImageFromArray(probmap_array)
                probmap_array.SetSpacing(global_inform['origin_spacing'])
                probmap_array.SetOrigin(global_inform['origin_origin'])
                probmap_array.SetDirection(global_inform['origin_direction'])

                sitk.WriteImage(probmap_array, os.path.join(store_path, "clabel.nii.gz"))

                if self.image_record_type == 1:
                    fnamelist = os.listdir(data_path)
                    if "mask.nii.gz" in fnamelist:
                        os.symlink(os.path.join(data_path, "mask.nii.gz"), os.path.join(store_path, 'mask.nii.gz'))
                    os.symlink(global_inform['origin_image_path'], os.path.join(store_path, 'im.nii.gz'), )
                elif self.image_record_type == 2:
                    fnamelist = os.listdir(data_path)
                    if "mask.nii.gz" in fnamelist:
                        os.link(os.path.join(data_path, "mask.nii.gz"), os.path.join(store_path, 'mask.nii.gz'))
                    os.link(global_inform['origin_image_path'], os.path.join(store_path, 'im.nii.gz'))

        if processing_id == 0:
            update_n = len(self.data_tag_list) - message_q.qsize() - bar.n
            bar.update(update_n)
            bar.close()

        logger.warning('Process:{}, Done!'.format(processing_id))
        return recorder

    def dataload(self, fold_path, global_inform):
        '''Data load
        Arg:
            fold_path: str, sample path
            global_inform: dict, contains 'tag'
        Return:
            output: dict, 'image' must|'label' optional, sitk.Image
            global_inform: dict, adds 'origin_spacing', 'origin_image_path', 'origin_shape'
        '''
        output = {}
        output['image'] = sitk.ReadImage(os.path.join(fold_path, 'im.nii.gz'))
    
        global_inform['origin_origin'] = output['image'].GetOrigin()
        global_inform['origin_direction'] = output['image'].GetDirection() 
        global_inform['origin_spacing'] = output['image'].GetSpacing()
        global_inform['origin_image_path'] = os.path.join(fold_path, 'im.nii.gz')
        origin_image = sitk.GetArrayFromImage(output['image'])
        global_inform['origin_shape'] = origin_image.shape

        if self.target_spacing != None:
            size = np.array(global_inform['origin_shape'])[::-1]
            spacing = np.array(global_inform['origin_spacing'])

            new_spacing = np.array(self.target_spacing)
            new_size = size * spacing / new_spacing
            new_spacing = size * spacing / new_size

            new_spacing = [float(s) for s in new_spacing]
            new_size = [int(round(s + 1e-4)) for s in new_size]
            global_inform['origin_shape'] = tuple(new_size[::-1])
            global_inform['origin_spacing'] = tuple(new_spacing)

        if self.compute_metric:
            output['label'] = sitk.ReadImage(os.path.join(fold_path, 'mask.nii.gz'))

        return output, global_inform

    def data_preprocess(self, data, global_inform):
        '''Data preprocess
        Args:
            data: dict, contains 'image' and others, sitk.Image
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape' and others
        Return:
            output: dict, contains 'image', padded image with shape (C, Z, Y, X)
            global_inform: dict, adds 'pad_s', 'padded_shape', 'ROI_inform'
        '''
        image = ImageResample(data['image'], self.spacing)
        image = sitk.GetArrayFromImage(image)

        image, pads = self._Padding(image, image.min())
        global_inform['pad_s'] = pads
        global_inform['padded_shape'] = image.shape

        ROI_se = []
        for _, l in enumerate(image.shape):
            ROI_se.append((0, l))
        global_inform['ROI_inform'] = ROI_se

        image = np.expand_dims(image, 0)
        output = {'image': image}

        return output, global_inform

    def patch_extract(self, data, global_inform):
        '''Patch extract strategy
        Args:
            data: dict, contains 'image' and others, np.Array
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape', 'ROI_inform', 'pad_s', 'padded_shape' and others
        Return:
            global_inform: dict, adds 'start_point', list [(zs,ys,xs)]
        '''
        ROI_inform = global_inform['ROI_inform']
        
        s = [[], [], []]
        for i, se in enumerate(ROI_inform):
            sp, ep = se
            l = ep - sp - self.patchsize[i]

            s[i].append(sp)
            if l == 0:
                continue

            s[i].append(ep - self.patchsize[i])
            if l == 1:
                continue

            if (ep - sp) <= (2 * self.patchsize[i]):
                s[i].append((sp + ep - self.patchsize[i]) // 2)
            else:
                spp = sp + self.patchsize[i] // 2
                while spp < (ep - self.patchsize[i]):
                    s[i].append(spp)
                    spp += self.patchsize[i] // 2

        start_point = []
        for zs in s[0]:
            for ys in s[1]:
                for xs in s[2]:
                    start_point.append((zs, ys, xs))

        global_inform['start_point'] = start_point

        return global_inform

    def generate_probmap(self, data, model_list, global_inform):
        '''Generate probability map
        Args:
            data: dict, contains 'image', padded image with shape (C, Z, Y, X)
            model_list: model list, [model1, model2], all on cpu
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape', 'ROI_inform', 'pad_s', 'padded_shape', 'start_point' and others
        Return:
            output: Tensor, probalility map
        '''
        processed_image = data['image'].astype('float32')
        image = torch.from_numpy(processed_image)
        start_points = global_inform['start_point']

        model_output_list = []
        model_cam_list = []
        for i in range(len(model_list)):
            model_list[i].to(device=global_inform['device'])
            for j in range(len(self.model_path[i])):
                params = self._load_params(self.model_path[i][j], device=global_inform['device'])
                model_list[i].load_state_dict(params)
                model_list[i].eval()

                output_list = []
                for start_point_batch in self._start_pointBatchgenerator(start_points):
                    patch_batch = self._startpoint2TensorBatch(image, start_point_batch)
                    output = self.forward_step((patch_batch, start_point_batch), model_list[i])
                    output_list.append(output)

                reconstructed_predict = self.label_reconstruction(output_list, global_inform)
                model_output_list.append(reconstructed_predict[0])
                model_cam_list.append(reconstructed_predict[1])

            model_list[i].to(device='cpu')
            torch.cuda.empty_cache()
        
        output = sum(model_output_list) / len(model_output_list)
        cam = sum(model_cam_list) / len(model_cam_list)

        return output, cam

    def _start_pointBatchgenerator(self, start_point_list):
        start = 0
        end = start + self.batchsize
    
        while start < len(start_point_list):
            start_point_batch = start_point_list[start:end]
            yield start_point_batch
            start = end 
            end = start + self.batchsize

    def _startpoint2TensorBatch(self, image, start_point_batch):
        batch_list = []
        lz, ly, lx = self.patchsize
        for sz, sy, sx in start_point_batch:
            patch = image[:, sz:(sz+lz), sy:(sy+ly), sx:(sx+lx)].unsqueeze(0)
            batch_list.append(patch)

        batch_list = torch.cat(batch_list, dim=0)

        return batch_list

    def forward_step(self, input_data, model):
        '''Forward step
        Args:
            input_data: tuple, (Image_Batch, start_point)
            model: nn.Module
        Return:
            output_result: dict, must contain normalized output of model 'output'
                and 'start_point'
        '''
        x, start_point = input_data
        x = x.cuda(non_blocking=True)
        output, cam = model(x)
        output = torch.softmax(output, dim=1)

        # cam = cam[:,1,:,:,:] - cam[:,0,:,:,:]

        camMin = torch.min(cam)
        camMax = torch.max(cam)

        cam = (cam - camMin) / (camMax - camMin)
        if self.memory_opt_level == 2 or self.memory_opt_level == 3:
            output = output.cpu()
            cam = cam.cpu()

        output_result = {'output': (output, cam), 'start_point': start_point}

        return output_result
    
    def label_reconstruction(self, batch_list, global_inform):
        '''Label reconstruct
        Args:
            batch_list: list, contains each step outputs of forward_step 
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape', 'ROI_inform', 'pad_s', 'padded_shape', 'start_point' and others
        Return:
            label: Tensor, reconstructed probability map, (N, C, Z, Y, X)
        '''
        padded_shape = global_inform['padded_shape']
        label = torch.zeros((len(self.class_tag), *padded_shape), dtype=torch.float).cuda()
        lz, ly, lx = self.patchsize

        prob = None
        for batch in batch_list:
            start_points = batch['start_point']
            prob = batch['output'][0].type_as(label)
            cam = batch['output'][1].type_as(label)
            for i, s in enumerate(start_points):
                zs, ys, xs = s
                label[:, zs:(zs+lz), ys:(ys+ly), xs:(xs+lx)] += cam[i]

        norm = label.sum(dim=0, keepdim=True) / len(self.class_tag)
        norm = torch.clamp_min(norm, 1)
        label /= norm

        c, z, y, x = label.size()
        pad_s = global_inform['pad_s']
        label = label[
            :,
            pad_s[0][0]:(z - pad_s[0][1]),
            pad_s[1][0]:(y - pad_s[1][1]),
            pad_s[2][0]:(x - pad_s[2][1]),
        ]

        if self.memory_opt_level == 1 or self.memory_opt_level == 3:
            label = label.cpu()

        return prob.squeeze(), label

    @abc.abstractmethod
    def postprocess(self, predict, global_inform):
        '''postprocess
        Args:
            predict: Tensor, reconstructed probability map, (N, C, Z, Y, X),
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape', 'ROI_inform', 'pad_s', 'padded_shape', 'start_point' and others
        Return:
            predict: tensor, resampled label, (C, Z, Y, X)
        '''
        origin_shape = global_inform['origin_shape']

        predict = F.interpolate(predict.cuda(), size=origin_shape, mode='trilinear', align_corners=True)

        return predict.squeeze(0)

    def _metric_compute(self, predict, label, recorder, global_inform):
        recorder['data_tag'].append(global_inform['tag'])

        predict_onehot = self._One_hot(predict)
        label_onehot = self._One_hot(label)
        for i, class_tag in enumerate(self.class_tag):
            if class_tag == 'background':
                continue

            metric = self.metric_compute(predict_onehot[i], label_onehot[i], global_inform)

            for j, v in enumerate(metric):
                recorder['{}_{}'.format(class_tag, self.metrics_tag[j])].append(v)

        return recorder

    def metric_compute(self, predict, label, global_inform):
        '''Metric compute
        Args:
            predict: np.Array, predict label, (Z, Y, X)
            label: np.Array, label, (Z, Y, X)
            global_inform: dict, contains 'tag', 'origin_spacing', 'origin_image_path', 
                'origin_shape', 'ROI_inform', 'pad_s', 'padded_shape', 'start_point' and others
        Return:
            metric: list, 1-d vector contains metrics as the sequence of self.metrics_tag
        '''

        return []

    def _load_params(self, mpath, device):
        result = torch.load(mpath, map_location=device)
        params = result['param']

        return params

    def _One_hot(self, label):
        onehot_label = np.eye(len(self.class_tag))[label]
        onehot_label = np.moveaxis(onehot_label, 3, 0)

        return onehot_label

    def _Padding(self, image, pad_val):
        pad_s = []
        for i, v in enumerate(image.shape):
            r = self.patchsize[i] - v
            if r > 0:
                pad_s.append((int(r / 2), r - int(r / 2)))
            else:
                pad_s.append((0, 0))

        image = np.pad(image, tuple(pad_s), mode='constant', constant_values=pad_val)

        return image, pad_s 

    def _recorder_init(self):
        recorder = OrderedDict()
        recorder['data_tag'] = []

        for class_name in self.class_tag:
            if class_name == 'background':
                continue

            for metric in self.metrics_tag:
                key = '{}_{}'.format(class_name, metric)
                recorder[key] = []

        return recorder

    def _logger_init(self, process_id):
        logger = logging.getLogger(self.model_tag + str(process_id))
        logger.setLevel(logging.INFO)

        ch1 = logging.StreamHandler()
        ch1.setLevel(logging.WARNING)
        ch1.setFormatter(logging.Formatter('%(name)s-Info:%(message)s'))
        logger.addHandler(ch1)

        # ch2 = logging.FileHandler(self.log_path.split('.')[0]+'{}.log'.format(process_id), 'w')
        # ch2.setLevel(logging.DEBUG)
        # ch2.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(ch2)

        # ch3 = logging.FileHandler(self.log_path, 'w')
        # ch3.setLevel(logging.INFO)
        # ch3.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(ch3)

        return logger        

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

        margin_list = []
        for i in range(label.ndim):
            edge_view = np.swapaxes(label, 0, i)
            l = edge_view.shape[0]
            edge_view = edge_view.reshape((l, -1)).sum(axis=1)
            lower, upper = findMargin(list(edge_view))

            margin_list.append((lower, upper))

        return margin_list