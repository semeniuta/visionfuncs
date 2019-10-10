import cv2
import numpy as np


def e2h(x):
    """
    Transform a Euclidean vector to homogeneous form
    """

    return np.array([el for el in x] + [1.0])


def h2e(x):
    """
    Transform a homogeneous vector to Euclidean form
    """

    return x[:-1] / x[-1]


def curvature_poly2(coefs, at_point):
    """
    Measure curvature of a second-order polynomial at the given point
    """

    a, b, _ = coefs
    return ((1 + (2 * a * at_point + b) ** 2) ** 1.5) / np.abs(2 * a)


def curvature_poly2_in_meters(coefs, at_point, meters_in_pix_x, meters_in_pix_y):
    """
    Curvature calculation based on polynomial
    coefficients estimated from pixel points
    """

    a, b, _ = coefs

    m_a = meters_in_pix_x / (meters_in_pix_y**2) * a
    m_b = (meters_in_pix_x / meters_in_pix_y) * b

    return ((1 + (2 * m_a * at_point + m_b) ** 2) ** 1.5) / np.abs(2 * m_a)


def pixel_points_to_meters(points, meters_in_pix_x, meters_in_pix_y):

    res = np.zeros_like(points, dtype=np.float32)
    res[:, 0] = points[:, 0] * meters_in_pix_x
    res[:, 1] = points[:, 1] * meters_in_pix_y

    return res


def rvec_to_rmat(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat


def triangulate_points(P1, P2, points1, points2):
    """
    Triangulate 3D coordinates of points
    based on point matches in two images.
    
    P1 - projection matrix of camera 1.
    P2 - projection matrix of camera 2.
    points1, points2 - arrays of corresponding
      points im image 1 and 2 respectively,
      each with dimension of (n x 2), 
      where n is the number of points.
    """

    assert points1.shape[0] == points2.shape[0]
    assert points1.shape[1] == 2 and points2.shape[1] == 2

    ptcloud_h = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    
    ptcloud = np.zeros((points1.shape[0], 3))

    for i in range(3):
        ptcloud[:, i] = ptcloud_h[i, :] / ptcloud_h[-1, :]
    
    return ptcloud


def theta_rho(line):
    """
    Given a homogenous vector representing a line
    returns a tuple (cos(theta), sin(theta), rho)
    for a line represented in a form
    rho = x*cos(theta) + y*sin(that)
    """

    norm = np.linalg.norm(line[:2])
    return np.array([line[0]/norm, line[1]/norm, -line[2]/norm])


def hnormalize(p):
    """
    Normalize a homogeneous vector p so that p[-1] = 1.
    Works also for vectors horizontally stacked in a matrix
    (where each vector is a column)
    """

    if p.ndim == 1:
        res = p / p[-1]
    else:
        res = np.ones_like(p)
        for j in range(p.shape[1]):
            res[:2, j] = p[:2, j] / p[2, j]
    
    return res