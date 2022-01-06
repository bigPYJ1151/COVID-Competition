
from re import L
import SimpleITK as sitk 
import numpy as np
from numpy.testing._private.utils import assert_array_equal, assert_equal

def StoicDataPreprocess(sitkImage):
    '''
    clip:[-1500,2500]
    spacing:[x,y,z][x,y,2.0]
    patchSize:[z,y,x][160,512,512]
    '''
    clipMin = -1500
    clipMax = 2500
    patchSize = (160,256,256)

    spacing = sitkImage.GetSpacing()
    newSpacing = [spacing[0] / 2, spacing[1] / 2, 2.0]
    sitkImage = ImageResample(sitkImage, newSpacing, False) 

    image = sitk.GetArrayFromImage(sitkImage)
    image = Padding(image, patchSize, clipMin)
    image = CentralCrop(image, patchSize)
    image = ClipandNormalize(image, clipMin, clipMax)

    return image

def Padding(imageArray:np.ndarray, size, padVal):
    assert_equal(np.ndim(imageArray), 3)

    pad_s = [] 
    for i, v in enumerate(imageArray.shape):
        r = size[i] - v
        if r > 0:
            pad_s.append((int(r / 2), r - int(r / 2)))
        else:
            pad_s.append((0, 0)) 

    return np.pad(imageArray, tuple(pad_s), mode='constant', constant_values=padVal)

def CentralCrop(imageArray:np.ndarray, size):
    assert_equal(np.ndim(imageArray), 3)
    dims = 3
    originSize = imageArray.shape

    dimRanges = []
    for i in range(dims):
        if originSize[i] > size[i]:
            mid = originSize[i] // 2
            dimRanges.append((mid - size[i] // 2, mid + size[i] // 2))
        else:
            dimRanges.append((0, originSize[i]))
        
        assert_equal(dimRanges[i][1] - dimRanges[i][0], size[i])

    return imageArray[
        dimRanges[0][0]:dimRanges[0][1],
        dimRanges[1][0]:dimRanges[1][1],
        dimRanges[2][0]:dimRanges[2][1],
    ]

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
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

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
