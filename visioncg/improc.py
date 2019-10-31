import cv2
import numpy as np


def grayscale(im, flag=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(im, flag)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def add_contrast(im, gain):
    """
    Add contrast to an image with the given gain.
    The resulting image is scaled back to the [0, 255] range
    """

    gained = gain * im
    return scale_image_255(gained)


def invert(image):
    return 255 - image
    

def clamp(image, a, b):
    """
    Clamp the range of intesities of the image 
    from (0, 255) to a custom range (a, b).
    """

    interval_len = b - a    
    return (interval_len / 255.0) * image + a


def scale_image_255(im):
    """
    Scale an image to pixel range [0, 255]
    """

    return np.uint8(255 * (im / np.max(im)))


def weighted_sum_images(images, weights):
    """
    Perfrom a weighted sum of 2 or more images
    """

    assert len(weights) == len(images)

    nonzero_indices = np.nonzero(weights)[0]
    if len(nonzero_indices) < 2:
        raise Exception('At least 2 non-zero weights are required')

    first, second = nonzero_indices[:2]
    res = cv2.addWeighted(images[first], weights[first], images[second], weights[second], 0)

    if len(nonzero_indices) == 2:
        return res

    for i in nonzero_indices[2:]:
        res = cv2.addWeighted(res, 1., images[i], weights[i], 0)

    return res


def dilate(im, kernel_size, n_iter=1):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(im, kernel, iterations=n_iter)


def image_histogram(im, n_bins=32, as_float=False):
    """
    Compute image histogram and return 
    the per-bin counts as an 1D array
    """

    counts, _ = np.histogram(im, bins=n_bins, range=(0, 256))
    
    if as_float:
        return np.array(counts, dtype=np.float64)
    
    return np.array(counts)


def image_is_dark(im, share=0.95):
    """
    Simple test for a grayscale image
    being mostly dark/black.
    """

    n_pixels = im.shape[0] * im.shape[1]
    hist = image_histogram(im)
    return hist[0] >= share * n_pixels