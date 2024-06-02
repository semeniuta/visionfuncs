import context as _
import skimage
from matplotlib import pyplot as plt

from visionfuncs import improc
from visionfuncs import circles

im = skimage.data.coins()
im_inverted = improc.invert(im)
blobs = circles.detect_circular_blobs(im_inverted)

_, (ax_original, ax_inverted) = plt.subplots(1, 2)
ax_original.imshow(im, cmap='gray')
ax_inverted.imshow(im_inverted, cmap='gray')

for blob in blobs:
    x, y = blob.pt
    radius = blob.size / 2.
    circle = plt.Circle((x, y), radius, edgecolor='yellow', fill=False)
    ax_inverted.add_artist(circle)

plt.show()
