import cv2
import numpy as np


def harris(im, block_size, ksize, k):    
    """
    block_size - neighborhood size
    ksize - aperture parameter for the Sobel operator
    k - Harris detector free parameter
    """
    
    return cv2.cornerHarris(im, block_size, ksize, k)


def harris_centroids(harris_dst, ratio_from_max):
    """
    Given the desitination image from cv2.cornerHarris,
    determine coordinates of the corners' centroids.
    
    ratio_from_max - a number between 0 and 1, multiplied with
    the maximal intesity of the image to determine the threshold
    """
    
    im = cv2.dilate(harris_dst, None)
    
    t = ratio_from_max * im.max()
    _, im = cv2.threshold(im, t, 255, 0)
    
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(im))
    
    return centroids