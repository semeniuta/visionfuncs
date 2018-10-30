import init
import sys
import os
import cv2

from rpa.io import open_image, sorted_glob

def read_calibration_images():

    imfiles1 = sorted_glob(os.path.join(init.CODE_DIR, 'DATA/IMG/calib/opencv_left/*.jpg'))
    imfiles2 = sorted_glob(os.path.join(init.CODE_DIR, 'DATA/IMG/calib/opencv_right/*.jpg'))

    images1 = [open_image(f, cv2.IMREAD_GRAYSCALE) for f in imfiles1]
    images2 = [open_image(f, cv2.IMREAD_GRAYSCALE) for f in imfiles2]

    return images1, images2