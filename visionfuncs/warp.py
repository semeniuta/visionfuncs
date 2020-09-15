import cv2
import numpy as np


def get_rectangle_corners_from_cbc(cbc, pattern_size_wh):
    """
    Get 4 points eclosing the chessboard region in an image

    cbc -- a (n x 2) NumPy array with each chessboard corner as a row
    pattern_size_wh -- a tuple (nx, ny) with the number of corners in x and y direction

    Returns a (4 x 2) matrix with each point as a row:
    [top left    ]
    [top right   ]
    [bottom right]
    [bottom left ]
    """

    nx, ny = pattern_size_wh

    points = np.array([
        cbc[0,:],
        cbc[nx-1,:],
        cbc[-1,:],
        cbc[nx*ny-nx,:],
    ], dtype=np.float32)

    return points


def get_rectangle_corners_in_image(warped_sz, offset_x, offset_y):
    """
    Get 4 points describing a rectangle in the image, offsetted
    by the given amounts from the edges, 
    along with the size of the resulting warted image.

    warped_sz -- size of the target rectangular region 
    offset_x, offset_y -- offsets in pixels from the edges of the image

    Returns a (4 x 2) matrix with each point as a row:
    [top left    ]
    [top right   ]
    [bottom right]
    [bottom left ]
    and the size of the resulting image, accounting for the offsets.
    """

    warped_x, warped_y = warped_sz

    warped_canvas_sz = (warped_x + 2 * offset_x, warped_y + 2 * offset_y)
    cols, rows = warped_canvas_sz

    points = np.array([
        [offset_x, offset_y],
        [cols-offset_x, offset_y],
        [cols-offset_x, rows-offset_y],
        [offset_x, rows-offset_y]
    ], dtype=np.float32)

    return points, warped_canvas_sz


def cb_dim_proportion(pattern_size_wh, factor=1):
    """
    Given the chessboard pattern size pattern_size_wh = (nx, ny), 
    i.e. number of corners in horizonal and vertical direction, 
    return the proportion (h, v) of the number of squares
    in horizontal and vertical direction, multiplied with 
    an optional factor. 
    """
        
    numbers_of_squares = (n_points - 1 for n_points in pattern_size_wh)
    
    return tuple(n_sq * factor for n_sq in numbers_of_squares)


def warp(im, M, canvas_sz):
    """
    Warp an image im given the perspective transformation matrix M and
    the output image size canvas_sz (cols, rows)
    """

    return cv2.warpPerspective(im, M, canvas_sz, flags=cv2.INTER_LINEAR)