from __future__ import print_function

import cv2
import numpy as np
from miptcv_utils import *


path = 'week#2/traffic_sign.jpg'

original = cv2.imread(path)
assert original is not None

img = cv2.resize(original, None, fx=0.25, fy=0.25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img)

img_vec = img.reshape((-1, 3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            100, 1.0)

n_pix = img.shape[0] * img.shape[1]

n_clusters = 256
ret, label, center = cv2.kmeans(
    img_vec, n_clusters, criteria, attempts=1,
    flags=cv2.KMEANS_RANDOM_CENTERS)

center = center.astype(np.uint8)
quant = center[label.flatten()].reshape(
    img.shape)

imshow_pair(img, quant)
