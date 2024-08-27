import context as _

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from visionfuncs import io
from visionfuncs import cbcalib
from visionfuncs import geometry


# from https://github.com/semeniuta/subset-calib-data
IMAGES_DIR = os.path.join(os.path.abspath(".."), "subset-calib-data", "data", "images")
PATTERN_SIZE = (9, 7)
SQUARE_SIZE = 20.0
IMAGES_SIZE = (1360, 1024)


def read_images(fnames, take_n):
    return tuple(
        io.open_image(f, read_flag=cv2.IMREAD_GRAYSCALE) for f in fnames[:take_n]
    )


if __name__ == "__main__":

    # Calibrate

    imfiles_1 = io.sorted_glob(os.path.join(IMAGES_DIR, "img_1_*.jpg"))
    imfiles_2 = io.sorted_glob(os.path.join(IMAGES_DIR, "img_2_*.jpg"))

    images_1 = read_images(imfiles_1, take_n=20)
    images_2 = read_images(imfiles_2, take_n=20)

    corners_1, corners_2, _ = cbcalib.prepare_corners_stereo(
        images_1, images_2, PATTERN_SIZE
    )

    pattern_points = cbcalib.prepare_object_points(20, PATTERN_SIZE, SQUARE_SIZE)

    _, cm_1, dc_1, rvecs_1, tvecs_1 = cbcalib.calibrate_camera(
        IMAGES_SIZE, pattern_points, corners_1
    )
    _, cm_2, dc_2, rvecs_2, tvecs_2 = cbcalib.calibrate_camera(
        IMAGES_SIZE, pattern_points, corners_2
    )

    R, T, E, F = cbcalib.calibrate_stereo(
        pattern_points, corners_1, corners_2, cm_1, dc_1, cm_2, dc_2, IMAGES_SIZE
    )

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cm_1, dc_1, cm_2, dc_2, IMAGES_SIZE, R, T
    )

    # Demo

    pattern_points = cbcalib.get_pattern_points(PATTERN_SIZE, SQUARE_SIZE)

    corners_1_ud = [
        cbcalib.undistort_points(src, cm_1, dc_1, P1, R1) for src in corners_1
    ]
    corners_2_ud = [
        cbcalib.undistort_points(src, cm_2, dc_2, P2, R2) for src in corners_2
    ]

    idx = 0
    projected_left = geometry.project_points(
        pattern_points, cm_1, dc_1, rvecs_1[idx], tvecs_1[idx]
    )
    projected_right = geometry.project_points(
        pattern_points, cm_2, dc_2, rvecs_2[idx], tvecs_2[idx]
    )

    triangulated = geometry.triangulate_points(
        P1, P2, corners_1_ud[idx], corners_2_ud[idx]
    )

    print("Distanced between the trinagulated points in the first row:")
    _, n_in_row = PATTERN_SIZE
    for i in range(n_in_row - 1):
        print("{:.3f}".format(np.linalg.norm(triangulated[i + 1] - triangulated[i])))

    fig, (ax_left, ax_right, ax_volume) = plt.subplots(1, 3)
    ax_left.imshow(images_1[idx], cmap="gray")
    ax_left.scatter(projected_left[:, 0], projected_left[:, 1], color="yellow", s=5)

    ax_right.imshow(images_2[idx], cmap="gray")
    ax_right.scatter(projected_right[:, 0], projected_right[:, 1], color="yellow", s=5)

    ax_volume = fig.add_subplot(1, 3, 3, projection="3d")
    ax_volume.scatter(triangulated[:, 0], triangulated[:, 1], triangulated[:, 2])
    ax_volume.set_xlabel("x")
    ax_volume.set_ylabel("y")
    ax_volume.set_zlabel("z")

    plt.show()
