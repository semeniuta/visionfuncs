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

    cg = cbcalib.CGCalibrateCamera()

    runner = compgraph.CompGraphRunner(cg, params)

    def closure(images):
        runner.run(calibration_images=images)
        return runner['camera_matrix'], runner['dist_coefs']

    return closure


def camera_calibration(images, psize, sqsize, im_wh):

    obj_points = cbcalib.prepare_object_points(len(images), psize, sqsize)
    img_points = cbcalib.prepare_corners(images, psize)

    _, camera_matrix, dist_coefs, _, _ = cbcalib.calibrate_camera(im_wh, obj_points, img_points)

    return camera_matrix, dist_coefs


if __name__ == '__main__':

    images1, images2 = read_calibration_images()

    cg_proc = get_proc_func_from_cg()
    direct_proc = lambda images: camera_calibration(images, PATTERN_SIZE, SQUARE_SIZE, IMAGE_WH)

    report = repeated_run(
        {'cg': cg_proc, 'direct': direct_proc},
        [images1],
        10
    )

    pprint(report)
    