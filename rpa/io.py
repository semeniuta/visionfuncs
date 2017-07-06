# Input/output for images and videos

import numpy as np
import PIL

def open_image(image_file, gray=True):
    ''' 
    Opens image file specified as a string
    '''

    im = PIL.Image.open(image_file)
    if gray:
        im = im.convert('L')
    return np.array(im)
