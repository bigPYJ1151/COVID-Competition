
import os 
from multiprocessing import Pool, cpu_count
import pickle
import SimpleITK as sitk 
import numpy as np
from torch import Assert
from .new_preprocess import preprocess
from tqdm import tqdm

def Resample(path_list, target_path, num_processes=None):
    '''
    Args:
        path_lisy:str
        target_path:str 
        spacing:(x,y,z)
        num_processs:defult=None
    '''
    if num_processes == None:
        num_processes = cpu_count()

    process_pool = Pool(num_processes)

    for i in range(num_processes):
        process_pool.apply_async(Image_resample_operation, args=(i, num_processes, path_list, target_path))

    process_pool.close()
    process_pool.join()

def Image_resample_operation(process_id, num_processes, path_list, target_path):
    print('Process {} launch.'.format(process_id))
    len_persplits = len(path_list) // num_processes

    start = len_persplits * process_id
    end = start + len_persplits

    if process_id == num_processes - 1:
        path_list = path_list[start:]
    else:
        path_list = path_list[start:end]

    if process_id == 0:
        path_list = tqdm(path_list)

    for source_path in path_list:
        fname = source_path.split('/')[-1]

        image = sitk.ReadImage(source_path)
        spacing = image.GetSpacing()
        image = preprocess(image, (1.0,1.0,2.0), (120,240,240))   # newpreprocess: default
                                    # newpreprocessv2: spacing:(1,1,2) patch:(120,240,240)
        Assert(image.shape == (120,240,240), "Size mismatch.")
        # image = sitk.GetImageFromArray(image)
        # image.SetSpacing(spacing)
        # sitk.WriteImage(image, os.path.join(target_path, fname))

        with open(os.path.join(target_path, fname.split('.')[0]), 'wb') as f:
            pickle.dump(image.astype('float16'), f)
    print('Process {} end.'.format(process_id))


def ImageResample(sitk_image, setting = [1.0, 1.0, 1.0], is_label = False, is_spacing=True):
    '''
    sitk_image:
    new_spacing: [x,y,z]
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())

    if is_spacing:
        new_spacing = np.array(setting)
        new_size = size * spacing / new_spacing
        new_spacing = size * spacing / new_size
    else:
        new_size = np.array(setting)
        new_spacing = size * spacing / new_size
        new_size = size * spacing / new_spacing
         
    new_spacing = [float(s) for s in new_spacing]
    new_size = [int(round(s + 1e-4)) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        #resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage
