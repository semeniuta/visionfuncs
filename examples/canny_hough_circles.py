import cv2
import skimage.data
import context as _
import skimage
from matplotlib import pyplot as plt

from visionfuncs import improc
from visionfuncs import circles


im = skimage.data.coins()
im_blurred = improc.gaussian_blur(im, kernel_size=13)
im_edges = cv2.Canny(im_blurred, threshold1=100, threshold2=150)
h_circles = circles.hough_circles(im_edges, dp=2, min_dist=4)

_, (ax_original, ax_blurred, ax_edges) = plt.subplots(1, 3)
ax_original.imshow(im, cmap='gray')
ax_blurred.imshow(im_blurred, cmap='gray')
ax_edges.imshow(im_edges)

for x, y, radius in h_circles:
    circle = plt.Circle((x, y), radius, edgecolor='yellow', fill=False)
    ax_original.add_artist(circle)

plt.show()
