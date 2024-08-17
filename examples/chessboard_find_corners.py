import context as _
import skimage
from matplotlib import pyplot as plt

from visionfuncs.warp import warp
from visionfuncs.cbcalib import find_corners_in_one_image


def assert_is_proportion(value):
    assert value >= 0.0 and value <= 1.0


def clip_image(im, keep_w=1.0, keep_h=1.0):

    assert_is_proportion(keep_w)
    assert_is_proportion(keep_h)

    h, w = im.shape

    offset_w = (w - int(w * keep_w)) // 2
    offset_h = (h - int(h * keep_h)) // 2

    return im[offset_h : h - offset_h, offset_w : w - offset_w]


if __name__ == "__main__":

    im = clip_image(skimage.data.checkerboard(), 0.95, 0.7)

    corners = find_corners_in_one_image(im, pattern_size_wh=(7, 5))

    plt.imshow(im, cmap="gray")
    plt.scatter(corners[:, 0], corners[:, 1], color="yellow")
    plt.show()
