import context as _
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt

from visionfuncs.warp import warp
from visionfuncs import geometry

np.set_printoptions(formatter={"float_kind": "{: .3f}".format})


def rotation_matrix(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, -s], [s, c]])


def create_transform(t_x, t_y, theta):

    translation = np.array([t_x, t_y])
    rotation = rotation_matrix(theta)

    transform = np.eye(3, dtype=float)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation

    return transform


if __name__ == "__main__":

    im = np.ones((300, 400, 3), dtype=np.uint8) * 255
    im[10:290, 10:390] = (0, 128, 255)
    cb = skimage.data.checkerboard()
    w, h = cb.shape
    top = 50
    left = 100

    for channel in range(3):
        im[top : top + w, left : left + h, channel] = cb

    src = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], dtype=np.float32)

    T_obj = create_transform(200, 150, 0)
    T = create_transform(0, 0, 0.2)

    src_t = geometry.transform_points(T_obj, src)
    dst = geometry.transform_points(T_obj @ T, src)

    M = cv2.getPerspectiveTransform(src_t, dst)
    im_warped = warp(im, M, (400, 300))

    print("T_obj = \n", T_obj)

    print("T = \n", T)

    print("T_obj @ T = \n", T_obj @ T)

    print("M = \n", M)

    _, (ax_original, ax_warped, ax_points) = plt.subplots(1, 3)
    ax_original.imshow(im, interpolation="none")
    ax_warped.imshow(im_warped, interpolation="none")
    ax_points.invert_yaxis()
    ax_points.axis("equal")
    ax_points.scatter(src_t[:, 0], src_t[:, 1])
    ax_points.scatter(dst[:, 0], dst[:, 1])
    plt.show()
