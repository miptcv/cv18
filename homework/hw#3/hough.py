from __future__ import print_function
from sys import argv
import cv2
import numpy as np


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    pass


def get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    pass


if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient,
                                           theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines,
                      min_delta_rho, min_delta_theta)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)
