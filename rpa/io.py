# Input/output for images and videos

import numpy as np
import PIL

def open_image(filename, gray=True):

    im = PIL.Image.open(filename)
    if gray:
        im = im.convert('L')
    return np.array(im)
