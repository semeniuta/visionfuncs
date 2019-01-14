def find_hough_lines(im_masked, rho, theta, threshold, min_line_length, max_line_gap):

    lines = cv2.HoughLinesP(im_masked, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines.reshape(lines.shape[0], 4)


def compute_line_tangents(lines):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    tans = (y2 - y1) / (x2 - x1)

    return tans


def line_vector_constant_y(val):
    return np.array([0, 1, -val])


def line_vector_from_opencv_points(line):

    x1, y1, x2, y2 = line
    return np.cross([x1, y1, 1], [x2, y2, 1])