import init
import sys
import os
import cv2
import time
from pprint import pprint

from rpa import cbcalib
from epypes import compgraph

from dataload import read_calibration_images
from experiment import repeated_run

IMAGE_WH = (640, 480)
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 10.   


def get_proc_func_from_cg():

    params = {
        'im_wh': IMAGE_WH,
        'pattern_size_wh': PATTERN_SIZE,
        'square_size': SQUARE_SIZE
    }

    cg = cbcalib.CGCalibrateStereo()

    runner = compgraph.CompGraphRunner(cg, params)

    def closure(images1, images2):
        runner.run(calibration_images_1=images1, calibration_images_2=images2)
        return runner['P1'], runner['P2']

    return closure


if __name__ == '__main__':

    images1, images2 = read_calibration_images()

    cg_proc = get_proc_func_from_cg()

    durations = repeated_run(
        {'cg': cg_proc},
        [images1, images2],
        10
    )

    pprint(durations)

    