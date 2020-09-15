import cv2
import numpy as np


def match_descriptors(descr_1, descr_2, normType, crossCheck):
    """
    Match keypoint descriptors using cv2.BFMatcher
    and return the matches sorted by distance.
    """

    matcher = cv2.BFMatcher(normType, crossCheck)
    matches = matcher.match(descr_1, descr_2)

    return sorted(matches, key=(lambda m : m.distance))


def gather_keypoints(keypoints_1, keypoints_2, matches):
    """
    Gather matched keypoints in a (n x 4) array,
    where each row correspond to a pair of matching 
    keypoints' coordinates in two images.
    """

    res = []

    for m in matches:

        idx_1 = m.queryIdx
        idx_2 = m.trainIdx

        pt_1 = keypoints_1[idx_1].pt
        pt_2 = keypoints_2[idx_2].pt

        row = [pt_1[0], pt_1[1], pt_2[0], pt_2[1]]
        res.append(row)

    return np.array(res)
