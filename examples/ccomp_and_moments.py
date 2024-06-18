import context as _
import numpy as np
import skimage
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from visionfuncs import regions


def plot_ellipse(ax, center_x, center_y, d1, d2, theta):

    ellipse = Ellipse([center_x, center_y], d1, d2, angle=math.degrees(theta), fill=False, color='black')
    ellipse.set_alpha(0.5)
    
    ax.add_artist(ellipse)


if __name__ == '__main__':

    generated_blobs = skimage.data.binary_blobs(volume_fraction=0.25, rng=42)
    im = np.array(generated_blobs, dtype=np.uint8)

    labels, ccomp_df = regions.find_ccomp(im)

    print(f'Connected components:\n{ccomp_df}')

    _, (ax_original, ax_ccomp) = plt.subplots(1, 2)
    ax_original.imshow(im, interpolation='none')
    ax_ccomp.imshow(labels, cmap='coolwarm', interpolation='none')        

    for value in range(1, len(ccomp_df)):

        row = ccomp_df.loc[value]
        ax_ccomp.text(row.x, row.y, value)

        component_im = regions.get_single_label_image(labels, value)
        center_x, center_y, d1, d2, theta = regions.region_ellipse_from_moments(component_im)
        plot_ellipse(ax_ccomp, center_x, center_y, d1, d2, theta)

    plt.show()
