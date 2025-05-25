import context as _
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt

from visionfuncs.warp import warp


def create_base_image(w, h, rim, backgroud_color=(0, 0, 255), top=50, left=100):

    im = np.ones((h, w, 3), dtype=np.uint8) * 255
    im[rim:h-rim, rim:w-rim] = backgroud_color
    cb = skimage.data.checkerboard()
    cb_w, cb_h = cb.shape

    for channel in range(3):
        im[top:top+cb_w, left:left+cb_h, channel] = cb

    return im


if __name__ == '__main__':

    w = 400
    h = 300
    rim = 10

    im = create_base_image(w, h, rim)

    last_x_before_rim = w - rim - 1
    last_y_before_rim = h - rim - 1

    src = np.array([
        [rim, rim], [last_x_before_rim, rim], [last_x_before_rim, last_y_before_rim], [rim, last_y_before_rim]
    ], dtype=np.float32)

    dst = np.array([
        [rim+30, rim], [last_x_before_rim-30, rim], [last_x_before_rim, last_y_before_rim], [rim, last_y_before_rim]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    im_warped = warp(im, M, (w, h))

    _, (ax_original, ax_warped) = plt.subplots(1, 2)
    ax_original.imshow(im, interpolation='none')
    ax_warped.imshow(im_warped, interpolation='none')
    plt.show()

