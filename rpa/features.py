import cv2
from epypes.compgraph import CompGraph, FunctionPlaceholder, add_new_vertices, graph_union_with_suffixing

METHOD_PARAMS = {
    'sift': ('nfeatures', 'nOctaveLayers', 'contrastThreshold', 'edgeThreshold', 'sigma'),
    'surf': ('hessianThreshold', 'nOctaves', 'nOctaveLayers', 'extended', 'upright'),
    'orb': ('nfeatures', 'scaleFactor', 'nlevels', 'edgeThreshold', 'firstLevel', 'WTA_K', 'scoreType', 'patchSize', 'fastThreshold')
}

METHOD_INIT = {
    'sift': cv2.xfeatures2d.SIFT_create,
    'surf': cv2.xfeatures2d.SURF_create,
    'orb': cv2.ORB_create
}

def create_concrete_feature_cg(method):

    if method not in METHOD_PARAMS:
        raise Exception('Feature detection/description method {} is not defined'.format(method))

    method_create = METHOD_INIT[method]
    param_names = METHOD_PARAMS[method]

    def func(*args):

        im, mask = args[:2]
        kvargs = {p_name: p_val for p_name, p_val in zip(param_names, args[2:]) if p_val != None}

        fd = method_create(**kvargs)

        return fd.detectAndCompute(im, mask)

    new_inputs = ('image', 'mask')  + param_names

    add_func_dict = {'detect_and_describe_features': func}
    add_func_io = {
        'detect_and_describe_features': (new_inputs, ('keypoints', 'descriptors'))
    }

    cg = add_new_vertices(CGFeatures(), add_func_dict, add_func_io)

    return cg

def create_feature_matching_cg(method):

    cg1 = create_concrete_feature_cg(method)
    cg2 = create_concrete_feature_cg(method)

    param_names = METHOD_PARAMS[method]

    def match_func(descr_1, descr_2, normType, crossCheck):

        matcher = cv2.BFMatcher(normType, crossCheck)
        return matcher.match(descr_1, descr_2)

    add_func_dict = {'match': match_func}
    add_func_io = {
        'match': (('descriptors_1', 'descriptors_2', 'normType', 'crossCheck'), 'matches')
    }

    cg = graph_union_with_suffixing(cg1, cg2, exclude=param_names)
    cg = add_new_vertices(cg, add_func_dict, add_func_io)

    return cg


class CGFeatures(CompGraph):

    def __init__(self):

        func_dict = {
            'detect_and_describe_features': FunctionPlaceholder()
        }

        func_io = {
            'detect_and_describe_features': (('image', 'mask'), ('keypoints', 'descriptors'))
        }

        super(CGFeatures, self).__init__(func_dict, func_io)
