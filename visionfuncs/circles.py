import cv2

def hough_circles(image, dp, min_dist, **other_kwargs):
    """
    Detect circles using Hough transform. 
    Returns a (n x 3) array, where every row
    correspond to (x, y, radius) of a circle.
    If circle detection is unsuccessful, 
    returns None.

    dp - inverse ratio of the accumulator resolution to the image resolution.
    min_dist - minimum distance between the centers of the detected circles.
    """

    hc_res = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, min_dist, **other_kwargs)

    if hc_res is None:
        return None

    return hc_res[0] 


def detect_circular_blobs(image, circ_tol=0.2):
    """
    Finds circular blobs in a grayscale or binary image
    using OpenCV's SimpleBlobDetector tool.

    Blobs are groups of dark connected pixels.

    Returns a list of KeyPoint object.
    """

    p = cv2.SimpleBlobDetector_Params()
    p.minCircularity = 1. - circ_tol
    p.maxCircularity = 1. + circ_tol
    p.filterByCircularity = True
    
    det = cv2.SimpleBlobDetector_create(p)
    blobs = det.detect(image)
    
    return blobs