def draw_line(canvas_im, line, color=[255, 0, 0], thickness=2):

    x1, y1, x2, y2 = line
    cv2.line(canvas_im, (x1, y1), (x2, y2), color, thickness)


def draw_lines_on_image(canvas_im, lines, color=[255, 0, 0], thickness=2):

    for i in range(lines.shape[0]):
        draw_line(canvas_im, lines[i, :], color, thickness)

def plot_line(line, **kvargs):

    xs = [line[0], line[2]]
    ys = [line[1], line[3]]

    plt.plot(xs, ys, '-', **kvargs)


def plot_homogeneous_line_vector(vec, x_from, x_to, **kvargs):

    a, b, c = vec

    def line_func(x):
        return (-a * x - c) / b

    xs = np.arange(x_from, x_to)
    ys = line_func(xs)

    plt.plot(xs, ys, **kvargs)