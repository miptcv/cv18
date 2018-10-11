from __future__ import print_function

import cv2
import numpy as np
from miptcv_utils import *
import matplotlib.pyplot as plt

# path = 'week#3/shadow1.png'
path = 'week#3/text.png'

original = cv2.imread(path)
assert original is not None

scaled = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
imshow(scaled)


# simple binarization
thresh = 150 # 128
binary = np.zeros(scaled.shape)
binary[scaled > thresh] = 255

# actual_thr, binary = cv2.threshold(
#     scaled, thresh, 255, cv2.THRESH_BINARY)
imshow_pair(scaled, binary)


# Otsu thresholding
otsu_thr, otsu = cv2.threshold(
    scaled, 0, 255, cv2.THRESH_BINARY +
                    cv2.THRESH_OTSU)
print(otsu_thr)
imshow_pair(scaled, otsu)
# plt.hist(scaled.reshape(-1,), 255)

imshow(scaled)
hist, bins = np.histogram(scaled, range(256))
plt.plot(range(1, 255), hist[1:])

plt.axvline(otsu_thr, color='red')
