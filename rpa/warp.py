def get_rectangle_corners_from_cbc(cbc, nx, ny):
    """
    Get 4 points eclosing the chessboard region in an image

    cbc -- a (n x 2) NumPy array with each chessboard corner as a row
    nx, ny -- number of corners in x and y direction

    Returns a (4 x 2) matrix with each point as a row:
    [top left    ]
    [top right   ]
    [bottom right]
    [bottom left ]
    """

    points = np.array([
        cbc[0,:],
        cbc[nx-1,:],
        cbc[-1,:],
        cbc[nx*ny-nx,:],
    ], dtype=np.float32)

    return points


def get_rectangle_corners_in_image(im_sz, offset_x, offset_y):
    """
    Get 4 points describing a rectangle in the image, offsetted
    by the given amounts from the edges.

    im_sz -- image size (cols, rows)
    offset_x, offset_y -- offsets in pixels from the edges of the image

    Returns a (4 x 2) matrix with each point as a row:
    [top left    ]
    [top right   ]
    [bottom right]
    [bottom left ]
    """

    points = np.array([
        [offset_x, offset_y],
        [im_sz[0]-offset_x, offset_y],
        [im_sz[0]-offset_x, im_sz[1]-offset_y],
        [offset_x, im_sz[1]-offset_y]
    ], dtype=np.float32)

    return points


def warp(im, M, canvas_sz):
    """
    Warp an image im given the perspective transformation matrix M and
    the output image size canvas_sz
    """

    return cv2.warpPerspective(im, M, canvas_sz, flags=cv2.INTER_LINEAR)