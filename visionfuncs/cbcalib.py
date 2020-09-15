"""
Vision systems calibration using a chessboard calibration object
"""

import cv2
import numpy as np
from .geometry import rvec_to_rmat
from .geometry import triangulate_points

findcbc_flags = {
    'default': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    'at_or_fq': cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
}

def find_cbc(im, pattern_size_wh, searchwin_size=5, findcbc_flags=None):
    """
    Find chessboard corners in the given image using OpenCV
    """

    if findcbc_flags == None:
        res = cv2.findChessboardCorners(im, pattern_size_wh)
    else:
        res = cv2.findChessboardCorners(im, pattern_size_wh, flags=findcbc_flags)

    found, corners = res

    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corners, (searchwin_size, searchwin_size), (-1, -1), term)

    return res


def cbc_opencv_to_numpy(success, cbc_res):
    """
    Transform the result of OpenCV's chessboard corners detection
    to a numpy array of size (n_corners x 2). If corners were not
    identified correctly, the function returns None
    """

    if success:
        return cbc_res.reshape(-1, 2)
    else:
        return None


def find_corners_in_one_image(im, pattern_size_wh, searchwin_size=5, findcbc_flags=None):

    found, corners = find_cbc(im, pattern_size_wh, searchwin_size, findcbc_flags)
    return cbc_opencv_to_numpy(found, corners)


def prepare_corners(images, pattern_size_wh, searchwin_size=5, findcbc_flags=None):
    """
    Find chessboard corners in the supplied images.

    Returns `corners_list`, a list containing NumPy arrays (n_corners x 2) for images with successful
    corners detection and None for the unsuccessful ones
    """

    corners_list = []

    for i, im in enumerate(images):

        res = find_corners_in_one_image(im, pattern_size_wh, searchwin_size, findcbc_flags)
        corners_list.append(res)

    return corners_list


def prepare_corners_stereo(images1, images2, pattern_size_wh, searchwin_size=5, findcbc_flags=None):

    corners1 = prepare_corners(images1, pattern_size_wh, searchwin_size, findcbc_flags)
    corners2 = prepare_corners(images2, pattern_size_wh, searchwin_size, findcbc_flags)

    res1 = []
    res2 = []
    for c1, c2 in zip(corners1, corners2):
        if not ((c1 is None) or (c2 is None)):
            res1.append(c1)
            res2.append(c2)

    num_images = len(res1)

    return res1, res2, num_images


def calibrate_camera(im_wh, object_points, image_points):
    """
    Perform camera calibration using a set of images with the chessboard pattern

    image_points -- a list of chessboard corners shaped as NumPy arrays (n_points x 2)

    Returns a tuple as a result of the cv2.calibrateCamera function call,
    containing the following calibration results:
    rms, camera_matrix, dist_coefs, rvecs, tvecs
    """

    res = cv2.calibrateCamera(object_points, image_points, im_wh, None, None)
    return res


def calibrate_stereo(object_points, impoints_1, impoints_2, cm_1, dc_1, cm_2, dc_2, im_wh):

    res = cv2.stereoCalibrate(object_points, impoints_1, impoints_2, cm_1, dc_1, cm_2, dc_2, im_wh)

    R, T, E, F = res[-4:]

    return R, T, E, F


def prepare_object_points(num_images, pattern_size_wh, square_size):
    """
    Prepare a list of object points matrices
    """

    pattern_points = get_pattern_points(pattern_size_wh, square_size)
    return make_list_of_identical_pattern_points(num_images, pattern_points)


def make_list_of_identical_pattern_points(num_images, pattern_points):
    return [pattern_points for i in range(num_images)]


def get_pattern_points(pattern_size_wh, square_size):
    """
    Form a matrix with object points for a chessboard calibration object
    """

    pattern_points = np.zeros((np.prod(pattern_size_wh), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size_wh).T.reshape(-1, 2)
    pattern_points *= square_size
    return pattern_points


def solve_pnp_ransac(pattern_points, image_points, cam_matrix, dist_coefs,
                     use_extrinsic_guess=False, iter_count=100, reproj_err_threshold=8.0, confidence=0.99):

    return cv2.solvePnPRansac(pattern_points, image_points, cam_matrix, dist_coefs,
                              use_extrinsic_guess, iter_count, reproj_err_threshold, confidence)


def project_points(object_points, rvec, tvec, cm, dc):
    """
    Project points using cv2.projectPoints
    and reshape the result to (n_points, 2)
    """

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, cm, dc)
    return projected.reshape(-1, 2)


def reprojection_rms(impoints_known, impoints_reprojected):
    """
    Compute root mean square (RMS) error of points
    reprojection (cv2.projectPoints).

    Both input NumPy arrays should be of shape (n_points, 2)
    """

    diff = impoints_known - impoints_reprojected

    squared_distances = np.sum(np.square(diff), axis=1)
    rms = np.sqrt(np.mean(squared_distances))

    return rms


def reproject_and_measure_error(image_points, object_points, rvecs, tvecs, cm, dc):
    """
    Given a list of image points (a NumPy array per image) and 
    a list of known object points (a NumPy array per image),
    perform reprojection for each image 
    with the known camera intrinsics (cm, dc) and extrinsics (rvecs, tvecs), 
    and measure RMS reprojection error for all points in all images. 
    """
    
    reproj_list = []
    
    for ip, op, rvec, tvec in zip(image_points, object_points, rvecs, tvecs):

        ip_reprojected = project_points(op, rvec, tvec, cm, dc)
        reproj_list.append(ip_reprojected)
        
    reproj_all = np.concatenate(reproj_list, axis=0)
    original_all = np.concatenate(image_points, axis=0)
    
    rms = reprojection_rms(original_all, reproj_all)
    return rms


def triangulate_impoints(P1, P2, impoints_1, impoints_2):
    """
    Perform triangulation of image points for 
    each image pair and collect the resulting 
    3D point clouds in a list.
    """
    
    points_3d_list = []
    
    for imp_1, imp_2 in zip(impoints_1, impoints_2):

        points_3d = triangulate_points(P1, P2, imp_1, imp_2)
        points_3d_list.append(points_3d)
        
    return points_3d_list


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


def undistort_points(points, cm, dc, P_mat=None, R_mat=None):

    n_points = len(points)
    src = points.reshape((n_points, 1, 2))

    if P_mat is None:
       P_mat = cm

    dst = cv2.undistortPoints(src, cm, dc, P=P_mat, R=R_mat)

    return dst.reshape((n_points, 2))


def prepare_indices_stereocalib(corners1, corners2):
    """
    Return indices between 0 and num_images
    for which chessboard corners were detected
    in both left and right image (i.e. neither in corners1 
    nor in corners2 there is None at those indices).
    """

    indices = []

    idx = 0
    for c1, c2 in zip(corners1, corners2):
        
        if not ((c1 is None) or (c2 is None)):
            indices.append(idx)
        
        idx += 1

    return indices


def cb_row(corners, pattern_size_wh, row_idx):
    """
    Get a row of chessboard corners with index row_idx.

    If non-valid row_idx is provided, None is returned.
    """
    
    row_size, n_rows = pattern_size_wh
    
    if row_idx < 0 or row_idx >= n_rows:
        return None
    
    offset = row_idx * row_size
    
    return corners[offset:offset+row_size, :]


def cb_col(corners, pattern_size_wh, col_idx):
    """
    Get a column of chessboard corners with index col_idx.

    If non-valid col_idx is provided, None is returned.
    """
    
    n_cols, col_size = pattern_size_wh
    
    if col_idx < 0 or col_idx >= n_cols:
        return None
        
    return np.array([corners[i * n_cols + col_idx, :] for i in range(col_size)])


def cb_diag(corners, pattern_size_wh):
    """
    Get all diagonal chessboard corners
    from (0, 0) to (n_rows, n_rows).
    """
    
    row_size, n_rows = pattern_size_wh
    
    return np.array([corners[i * row_size + i] for i in range(n_rows)])


def corners_to_homog(corners):
    """
    Transform chessboard corners to homogeneous
    coordinates, with the matrix of shape
    (3, n) being returned.
    """
    
    n = len(corners)
    
    corners_with_one = np.ones((n, 3))
    corners_with_one[:, :2] = corners
    
    return corners_with_one.T