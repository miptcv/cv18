from __future__ import print_function

import cv2
import numpy as np
from miptcv_utils import *

path = 'week#3/shadow1.png'
original = cv2.imread(path)
assert original is not None


scaled = cv2.resize(original, None, fx=0.5, fy=0.5)
scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
# imshow(scaled)


def niblack(img, w, k):
    img  = img.astype(float)
    img2 = np.square(img)

    ave  = cv2.blur(img,  (w, w))
    ave2 = cv2.blur(img2, (w, w))

    n = np.multiply(*img.shape)
    std = np.sqrt((ave2 * n - img2) / n / (n-1))

    t = ave + k * std
    plt.imshow(t, cmap='gray')
    binary = np.zeros(img.shape)
    binary[img >= t] = 255
    return binary


inverted = 255 - scaled
binary_inv = niblack(inverted, 100, -0.2)
binary = 255 - binary_inv

imshow_pair(scaled, binary)
