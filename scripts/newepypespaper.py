import sys, os
CODE_DIR = os.environ['PHD_CODE']
sys.path.append(os.path.join(CODE_DIR, 'EPypes'))
sys.path.append(os.path.join(CODE_DIR, 'RPALib'))

import cv2

from epypes.compgraph import CompGraph, CompGraphRunner, get_networkx_graph
from networkx.drawing.nx_agraph import to_agraph


def grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


if __name__ == '__main__':

    func_dict = {
        'grayscale': grayscale,
        'canny': cv2.Canny,
        'blur': gaussian_blur
    }

    func_io = {
        'grayscale': ('image', 'image_gray'),
        'blur': (('image_gray', 'blur_kernel'), 'image_blurred'),
        'canny': (('image_blurred', 'canny_lo', 'canny_hi'), 'edges'),
    }

    cg = CompGraph(func_dict, func_io)

    params = {
        'blur_kernel': 11,
        'canny_lo': 70,
        'canny_hi': 200
    }

    runner = CompGraphRunner(cg, params)

    for k, obj in {'cg': cg, 'runner': runner}.items():

        nxg = get_networkx_graph(obj, style_attrs={'fontname': 'Helvetica'})

        ag = to_agraph(nxg)
        ag.layout('dot')
        ag.draw('simple_cg_{}.pdf'.format(k))
