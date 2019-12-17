import cv2
import numpy as np
from .geometry import hnormalize


def hough_lines(im_masked, rho, theta, threshold, min_line_length, max_line_gap):
    """
    rho - distance resolution of the accumulator in pixels.
    theta - angle resolution of the accumulator in radians.
    threshold - accumulator threshold parameter;
                only those lines are returned that get enough votes ( >threshold ).
    min_line_Length - minimum line length. Line segments shorter than that are rejected.
    max_line_gap – maximum allowed gap between points on the same line to link them.
    """

    lines = cv2.HoughLinesP(im_masked, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines.reshape(lines.shape[0], 4)


def compute_line_tangents(lines):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    tans = (y2 - y1) / (x2 - x1)

    return tans


def partition_lines(lines, tol=1e-3):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    xdiff = np.abs(x2 - x1)
    ydiff = np.abs(y2 - y1)

    h_mask = ydiff <= tol
    v_mask = xdiff <= tol
    rest_mask = np.logical_and(np.logical_not(h_mask), np.logical_not(v_mask))

    return lines[h_mask], lines[v_mask], lines[rest_mask]


def line_vector_constant_y(val):
    return np.array([0, 1, -val])


def line_vector_from_opencv_points(line):

    x1, y1, x2, y2 = line
    line_vec = np.cross([x1, y1, 1], [x2, y2, 1])

    return hnormalize(line_vec)