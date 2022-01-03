
import os 
import time

import numpy as np
import SimpleITK as sitk 

def ExecuteTransform(image, transform, is_label=False, pad_val=0):
    '''Execute transform
    Args:
        image: sitk.Image
        transform: sitk.Transform
        is_label: bool, default=False
        pad_val: int or float, default=0
    '''
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkBSpline
        # interpolator = sitk.sitkLinear

    return sitk.Resample(image, image, transform, interpolator, pad_val)

def Rotation3D(reference_image, angle=(0, 0, 0), center=(0, 0, 0)):
    '''Rotation transform on 3D image
    Arg:
        reference_image: sitk.Image, reference image
        angle: tuple, (angle_x, angle_y, angle_z), degree, counterclockwise +
        center: tuple, (c_x, c_y, c_z)
    '''
    angle = tuple(((i / 180) * np.pi for i in angle))
    center = reference_image.TransformContinuousIndexToPhysicalPoint(center)
    t = sitk.Euler3DTransform()
    t.SetRotation(*angle)
    t.SetCenter(center)

    return t

def TranslationandRotation3D(reference_image, angle=(0, 0, 0), center=(0, 0, 0), offset=(0, 0, 0)):
    '''Translation first, then rotation on 3D image
    Args:
        reference_image: sitk.Image, reference image
        angle: tuple, (angle_x, angle_y, angle_z), degree, counterclockwise +
        center: tuple, (c_x, c_y, c_z)
        offset:(x_offset, y_offset, z_offset)
    '''
    angle = tuple(((i / 180) * np.pi for i in angle))
    center = reference_image.TransformContinuousIndexToPhysicalPoint(center)
    offset = reference_image.TransformContinuousIndexToPhysicalPoint(offset)
    offset = tuple((-i for i in offset))
    t = sitk.Euler3DTransform()
    t.SetRotation(*angle)
    t.SetCenter(center)
    t.SetTranslation(offset)

    return t

def ComposeTransforms(transforms_list):
    '''Composing transforms in transforms_list sequally
    Args:
        transforms_list: list, [t1, t2, t3, ...]
    '''
    # transforms_list.reverse()
    t = sitk.Transform()

    for tc in transforms_list:
        t.AddTransform(tc)

    return t

def Translation3D(reference_image, offset=(0, 0, 0)):
    '''Translation transform on 3D image
    Args:
        reference_image: sitk.Image, reference image
        offset:(x_offset, y_offset, z_offset)
    '''
    offset = reference_image.TransformContinuousIndexToPhysicalPoint(offset)
    offset = tuple((-i for i in offset))
    return sitk.TranslationTransform(3, offset)

def Scale3D(reference_image, scale=(1, 1, 1), center=(0, 0, 0)):
    '''Scaling 3D image
    Args:
        reference_image: sitk.Image, reference image
        scale: tuple, (s_x, s_y, s_z)
        center: tuple, (c_x, c_y, c_z)
    '''
    scale = tuple((1.0 / i for i in scale))
    center = reference_image.TransformContinuousIndexToPhysicalPoint(center)
    t = sitk.ScaleTransform(3, scale)
    t.SetCenter(center)

    return t

def ElasticDefrom3D(reference_image, num_control_points=2, sigma=1, no_z=True):
    '''Elastic Deformation on 3D image
    Args:
        reference_image: sitk.Image, reference image
        num_control_points: int, number of defromation control point per axis, default=2
        sigma: int or float, variance of displacement, default=1
        no_z: bool, whether displacing on z axis, default=True
    '''
    num_control_points = max(num_control_points, 2)
    sigma = max(sigma, 1)

    t_mesh_size = [num_control_points] * 3
    t = sitk.BSplineTransformInitializer(reference_image, t_mesh_size)
    params = t.GetParameters()
    params = np.array(params, dtype=np.float)
    params += np.random.randn(params.shape[0]) * sigma
    num_params = params.shape[0] // 3 

    if no_z:
        params[2 * num_params:] = 0

    t.SetParameters(params)

    return t

# if __name__ == "__main__":
#     origin_image = sitk.ReadImage('image.nii.gz')
#     origin_label = sitk.ReadImage('label.nii.gz')

#     offset = (50, 0, 0)
#     angle=(0.0, 0, 0)
#     scale = (1, 1, 1)
#     print(origin_image.GetSize())
#     center = tuple(np.array(origin_image.GetSize()) / 2)
#     print(center)
#     start = time.time()

#     # transform = Translation3D(origin_image, offset)
#     # transform = Rotation3D(origin_image, angle=angle, center=center)
#     # transform = TranslationandRotation3D(origin_image, angle, center, offset)
#     # transform = ComposeTransforms([
#     #     Translation3D(origin_image, offset),
#     #     Rotation3D(origin_image, angle=angle, center=center),
#     # ])
#     # transform = Scale3D(origin_image, scale, center)
#     # transform = ComposeTransforms([
#     #     TranslationandRotation3D(origin_image, angle, center, offset),
#     #     Scale3D(origin_image, scale, center),
#     # ])

#     transform = ElasticDefrom3D(origin_image)

#     middle = time.time()

#     image = ExecuteTransform(origin_image, transform, is_label=False, pad_val=0)
#     label = ExecuteTransform(origin_label, transform, is_label=True, pad_val=0)

#     end = time.time()
#     print(middle - start)
#     print(end - middle)

#     sitk.WriteImage(image, 't_image.nii.gz')
#     sitk.WriteImage(label, 't_label.nii.gz')