# Vision systems calibration using a chessboard caloibratiob object

import cv2
import numpy as np
from epypes import compgraph

findcbc_flags = {
    'default': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    'at_or_fq': cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
}

def find_cbc(img, pattern_size_wh, searchwin_size=5, findcbc_flags=None):
    '''
    Find chessboard corners in the given image using OpenCV
    '''

    if findcbc_flags == None:
        res = cv2.findChessboardCorners(img, pattern_size_wh)
    else:
        res = cv2.findChessboardCorners(img, pattern_size_wh, flags=findcbc_flags)

    found, corners = res

    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (searchwin_size, searchwin_size), (-1, -1), term)

    return res


def cbc_opencv_to_numpy(success, cbc_res):
    '''
    Transform the result of OpenCV's chessboard corners detection
    to a numpy array of size (n_corners x 2). If corners were not
    identified correctly, the function returns None
    '''

    if success:
        return cbc_res.reshape(-1, 2)
    else:
        return None


def calibrate_camera(im_wh, object_points, image_points):
    '''
    Perform camera calibration using a set of images with the chessboard pattern

    image_points -- a list of chessboard corners shaped as NumPy arrays (n_points x 2)

    Returns a tuple as a result of the cv2.calibrateCamera function call,
    containing the following calibration results:
    rms, camera_matrix, dist_coefs, rvecs, tvecs
    '''

    res = cv2.calibrateCamera(object_points, image_points, im_wh, None, None)
    return res

def calibrate_stereo(object_points, impoints_1, impoints_2, cm_1, dc_1, cm_2, dc_2, im_wh):

    res = cv2.stereoCalibrate(object_points, impoints_1, impoints_2, cm_1, dc_1, cm_2, dc_2, im_wh)

    R, T, E, F = res[-4:]

    return R, T, E, F


def triangulate_points(P1, P2, points1, points2):

    points1_matrix = np.transpose(np.array(points1))
    points2_matrix = np.transpose(np.array(points2))

    res = cv2.triangulatePoints(P1, P2, points1_matrix, points2_matrix)
    res = np.transpose(res)

    #res_real = np.array([[row[i] / row[3] for i in range(3)] for row in res])
    #return res_real

    return res

def prepare_object_points(num_images, pattern_size_wh, square_size):
    '''
    Prepare a list of object points matrices
    '''

    pattern_points = get_pattern_points(pattern_size_wh, square_size)
    object_points = [pattern_points for i in range(num_images)]
    return object_points


def get_pattern_points(pattern_size_wh, square_size):
    '''
    Form a matrix with object points for a chessboard calibration object
    '''

    pattern_points = np.zeros((np.prod(pattern_size_wh), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size_wh).T.reshape(-1, 2)
    pattern_points *= square_size
    return pattern_points

def solve_pnp_ransac(pattern_points, image_points, cam_matrix, dist_coefs,
                     use_extrinsic_guess=False, iter_count=100, reproj_err_threshold=8.0, confidence=0.99):

    return cv2.solvePnPRansac(pattern_points, image_points, cam_matrix, dist_coefs,
                              use_extrinsic_guess, iter_count, reproj_err_threshold, confidence)

def rvec_to_rmat(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat

def get_im_wh(im):
    h, w = im.shape[:2]
    return w, h


def undistort_and_rectify_images_stereo(images1, images2, cm1, dc1, cm2, dc2, R1, R2, P1, P2):

    im_wh = get_im_wh(images1[0])

    maps1 = cv2.initUndistortRectifyMap(cm1, dc1, R1, P1, im_wh, m1type=cv2.CV_16SC2)
    maps2 = cv2.initUndistortRectifyMap(cm2, dc2, R2, P2, im_wh, m1type=cv2.CV_16SC2)

    interp_method = cv2.INTER_LINEAR

    images1_rect = [cv2.remap(im, maps1[0], maps1[1], interp_method) for im in images1]
    images2_rect = [cv2.remap(im, maps2[0], maps2[1], interp_method) for im in images2]

    return images1_rect, images2_rect, maps1, maps2

class CGFindCorners(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'find_corners': find_cbc,
            'reformat_corners': cbc_opencv_to_numpy
        }

        func_io = {
            'find_corners': (('image', 'pattern_size_wh'), ('success', 'corners_opencv')),
            'reformat_corners': (('success', 'corners_opencv'), 'corners_np')
        }

        super(CGFindCorners, self).__init__(func_dict, func_io)

class CGCalibrateCamera(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'prepare_corners': compgraph.FunctionPlaceholder(),
            'count_images': lambda lst: len(lst),
            'prepare_object_points': prepare_object_points,
            'calibrate_camera': calibrate_camera
        }

        func_io = {
            'prepare_corners': (('calibration_images', 'pattern_size_wh'), 'image_points'),
            'count_images': ('calibration_images', 'num_images'),
            'prepare_object_points': (('num_images', 'pattern_size_wh', 'square_size'), 'object_points'),
            'calibrate_camera': (('im_wh', 'object_points', 'image_points'),
                                 ('rms', 'camera_matrix', 'dist_coefs', 'rvecs', 'tvecs'))
        }

        super(CGCalibrateCamera, self).__init__(func_dict, func_io)

class CGSolvePnP(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'solve_pnp': cv2.solvePnP,
            'rvec_to_rmat': rvec_to_rmat
        }

        func_io = {
            'solve_pnp': (('pattern_points', 'image_points', 'cam_matrix', 'dist_coefs', 'use_extrinsic_guess'),
                          ('rvec', 'tvec')),

            'rvec_to_rmat': ('rvec', 'rmat')
        }

        super(CGSolvePnP, self).__init__(func_dict, func_io)

class CGCalibrateStereo(compgraph.CompGraph):

    def __init__(self):

        func_dict = {
            'prepare_corners': compgraph.FunctionPlaceholder(),
            'prepare_object_points': prepare_object_points,
            'calibrate_camera_1': calibrate_camera,
            'calibrate_camera_2': calibrate_camera,
            'calibrate_stereo': calibrate_stereo,
            'compute_rectification_transforms': cv2.stereoRectify
        }

        func_io = {
            'prepare_corners': (('calibration_images_1', 'calibration_images_2', 'pattern_size_wh'), ('image_points_1', 'image_points_2', 'num_images')),
            'prepare_object_points': (('num_images', 'pattern_size_wh', 'square_size'), 'object_points'),
            'calibrate_camera_1': (('im_wh', 'object_points', 'image_points_1'),
                                  ('rms_1', 'cm_1', 'dc_1', 'rvecs_1', 'tvecs_1')),
            'calibrate_camera_2': (('im_wh', 'object_points', 'image_points_2'),
                                   ('rms_2', 'cm_2', 'dc_2', 'rvecs_2', 'tvecs_2')),
            'calibrate_stereo' : (('object_points', 'image_points_1', 'image_points_2', 'cm_1', 'dc_1', 'cm_2', 'dc_2', 'im_wh'),
                                  ('stereo_rmat', 'stereo_tvec', 'essential_mat', 'fundamental_mat')),
            'compute_rectification_transforms': (('cm_1', 'dc_1', 'cm_2', 'dc_2', 'im_wh', 'stereo_rmat', 'stereo_tvec'),
                                                 ('R1', 'R2', 'P1', 'P2', 'Q', 'validPixROI1', 'validPixROI2'))
        }

        super(CGCalibrateStereo, self).__init__(func_dict, func_io)
