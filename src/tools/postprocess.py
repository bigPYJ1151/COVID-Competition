
import numpy as np 
from scipy.ndimage.measurements import label as scipy_label
from skimage.morphology import remove_small_objects

def maxConnectComponets(image, ignore_label=0):
    '''
    image:one-hot image
    return image:one-hot image
    '''

    for i in range(image.shape[0]):
        if i == ignore_label:
            continue

        image_c = image[i].astype('bool')
        image_c = remove_small_objects(image_c)
        image_labeled, n = scipy_label(image_c)

        label_area = []
        for j in range(n):
            label_area.append((image_labeled == (j + 1)).sum())

        label_area = np.array(label_area)
        if len(label_area) != 0:
            max_index = np.argmax(label_area)
            new_label = (image_labeled == (max_index + 1))
            
            image[ignore_label] = (image[ignore_label].astype('bool') | image_labeled.astype('bool')) & (~new_label)
            image[i] = new_label

    return image.astype('int')

def percentConnectComponents(image, percent=0.1, ignore_label=0):
    '''
    image:one-hot image
    return image:one-hot image
    '''
    if percent < 0 or percent > 1:
        raise Exception('Invalid percent!')

    for i in range(image.shape[0]):
        if i == ignore_label:
            continue

        image_c = image[i]
        volume = image_c.sum()
        image_c = image_c.astype('bool')
        threshold = int(volume * percent)
        image_c = remove_small_objects(image_c, min_size=threshold, connectivity=image_c.ndim)
        image[ignore_label] = image[ignore_label].astype('bool') | ((~image_c.astype('bool')) & image[i].astype('bool'))
        image[i] = image_c

    return image.astype('int')