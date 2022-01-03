
import SimpleITK as sitk 
import numpy as np

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

def ClipandNormalize(image_array : np.array, clip_min : int, clip_max : int):
    image_array = np.clip(image_array, clip_min, clip_max)
    num_seq = image_array[((image_array != clip_min) * (image_array != clip_max)).astype('bool')]
    mean_val = num_seq.mean()
    std_val = num_seq.std()
    
    return (image_array - mean_val) / std_val

def IIRGaussianSmooth(sitk_image, sigma):
    return sitk.SmoothingRecursiveGaussian(sitk_image, sigma, False)
