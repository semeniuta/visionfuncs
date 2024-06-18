import context as _
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt

from visionfuncs.warp import warp

if __name__ == '__main__':

    im = np.ones((300, 400, 3), dtype=np.uint8) * 255
    im[10:290, 10:390] = (0, 0, 255)
    cb = skimage.data.checkerboard()
    w, h = cb.shape
    top = 50
    left = 100

    for channel in range(3):
        im[top:top+w, left:left+h, channel] = cb

    src = np.array([
        [10, 10], [289, 10], [389, 289], [10, 289]
    ], dtype=np.float32)

    dst = np.array([
        [10+30, 10], [289-30, 10], [389, 289], [10, 289]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    im_warped = warp(im, M, (400, 300))

    _, (ax_original, ax_warped) = plt.subplots(1, 2)
    ax_original.imshow(im, interpolation='none')
    ax_warped.imshow(im_warped, interpolation='none')
    plt.show()

