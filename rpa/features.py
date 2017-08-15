import cv2
from epypes.compgraph import CompGraph

class CGFeaturesSIFT(CompGraph):

    def __init__(self):

        self._sift = cv2.xfeatures2d.SIFT_create()

        func_dict = {
            'detect_keypoints': self._sift.detect,
            'compute_descriptors': self._sift.compute
        }

        func_io = {
            'detect_keypoints': ('image', 'keypoints'),
            'compute_descriptors': (('image', 'keypoints'), 'descriptors')
        }

        super(CGFeaturesSIFT, self).__init__(func_dict, func_io)



