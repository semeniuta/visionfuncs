# Input/output for images and videos

import cv2
from glob import glob
import numpy as np


def open_image(fname, read_flag=cv2.IMREAD_COLOR, color_transform=None):
    """
    Open image usign OpenCV.

    Possible `read_flag`s are cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_ANYCOLOR.
    Example of a`color_transform` is cv2.COLOR_BGR2RGB
    """

    im = cv2.imread(fname, read_flag)

    if color_transform is not None:
        im = cv2.cvtColor(im, color_transform)

    return im


def sorted_glob(mask):
    return sorted(glob(mask))

