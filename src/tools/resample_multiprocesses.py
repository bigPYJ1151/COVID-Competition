
import os 
from multiprocessing import Pool, cpu_count
import SimpleITK as sitk 
import numpy as np
from tqdm import tqdm

def Resample(path_list, target_path, spacing, num_processes=None):
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
        process_pool.apply_async(Image_resample_operation, args=(i, num_processes, path_list, target_path, spacing))

    process_pool.close()
    process_pool.join()

def Image_resample_operation(process_id, num_processes, path_list, target_path, setting):
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

        if os.path.exists(os.path.join(target_path, fname)) == False:
            os.mkdir(os.path.join(target_path, fname))

        image = sitk.ReadImage(os.path.join(source_path, 'im.nii.gz'))
        image = ImageResample(image, setting, False)
        sitk.WriteImage(image, os.path.join(target_path, fname, 'im.nii.gz'))
        
        if os.path.exists(os.path.join(source_path, 'mask.nii.gz')) == True:
            label = sitk.ReadImage(os.path.join(source_path, 'mask.nii.gz'))
            label = ImageResample(label, setting, True)

            if image.GetSize() != label.GetSize():
                print('Alert, {}!!'.format(fname))

            sitk.WriteImage(label, os.path.join(target_path, fname, 'mask.nii.gz'))

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
