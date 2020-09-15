import cv2
import numpy as np
from .improc import scale_image_255


def sobel_x(im):
    return cv2.Sobel(im, cv2.CV_64F, 1, 0)


def sobel_y(im):
    return cv2.Sobel(im, cv2.CV_64F, 0, 1)


def sobel_abs(sobel):
    return scale_image_255(np.abs(sobel))


def sobel_magnitude(sobelx, sobely):
    return np.sqrt(np.square(sobelx) + np.square(sobely))


def sobel_magnitude_from_image(im):

    sobelx = sobel_x(im)
    sobely = sobel_y(im)

    return sobel_magnitude(sobelx, sobely)


def sobel_direction(sobelx, sobely):
    return np.arctan2(np.abs(sobely), np.abs(sobelx))