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

    x = np.array(x)
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