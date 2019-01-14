

def apply_region_mask(image, region_vertices):

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region_vertices, 255)

    return cv2.bitwise_and(image, mask)


def mask_threshold_range(im, thresh_min, thresh_max):
    """
    Return a binary mask image where pixel intensities
    of the original image lie within [thresh_min, thresh_max)
    """

    binary_output = (im >= thresh_min) & (im < thresh_max)
    return np.uint8(binary_output)


def bitwise_or(images):
    """
    Apply bitwise OR operation to a list of images
    """

    assert len(images) > 0

    if len(images) == 1:
        return images[0]

    res = cv2.bitwise_or(images[0], images[1])
    if len(images) == 2:
        return res

    for im in images[2:]:
        res = cv2.bitwise_or(res, im)

    return res