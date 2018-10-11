from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow_gray(image):
    plt.imshow(255 - image, cmap='gray')
    plt.xticks([]); plt.yticks([])


def imshow_pair(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(255- img1, cmap='gray')
    ax2.imshow(255- img2, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    f.show()


path = 'week#2/traffic_sign.jpg'

original = cv2.imread(path)
assert original is not None

img = cv2.resize(original, None, fx=0.25, fy=0.25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# H/V derivatives
def make_hv_diffs(img, kernel_x, kernel_y):
    ver_grad = cv2.filter2D(img, -1, kernel_x)
    hor_grad = cv2.filter2D(img, -1, kernel_y)
    imshow_pair(255 - ver_grad, 255 - hor_grad)


diff_x = np.array([[1, -1]])
diff_y = np.array([[1], [-1]])
make_hv_diffs(img, diff_x, diff_y)


# Robert Cross
robert_cross_x = np.array([[-1, 0],
                            [0, 1]])
robert_cross_y = np.array([[0, -1],
                           [1,  0]])
make_hv_diffs(img, robert_cross_x, robert_cross_y)


# Sobel
sobel_x = np.array(([-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]))
sobel_y = np.array(([-1, -2, -1],
                    [0,   0,  0],
                    [1,   2,  1]))
make_hv_diffs(img, sobel_x, sobel_y)

# Laplacian:
# [0,  1, 0]
# [1, -4, 1]
# [0,  1, 0]
lapl = cv2.Laplacian(img, -1)
imshow_gray(lapl)


# Box filter (integral image)
boxed = cv2.blur(img, (50, 50))
imshow_pair(boxed, img)


# Gaussian blur (separable)
smoothed = cv2.GaussianBlur(
    img, (0, 0), sigmaX=10, sigmaY=10)

smoothed_x = cv2.GaussianBlur(
    img, (0, 0), sigmaX=10, sigmaY=1)
smoothed_y = cv2.GaussianBlur(
    smoothed_x, (0, 0), sigmaX=1, sigmaY=10)
imshow_pair(smoothed, smoothed_y - smoothed)


# Difference of Gaussian
smoothed_1 = cv2.GaussianBlur(
    img, (0, 0), sigmaX=5, sigmaY=5)
smoothed_2 = cv2.GaussianBlur(
    img, (0, 0), sigmaX=4, sigmaY=4)
dog = smoothed_2 - smoothed_1
imshow_gray(dog)


# getGaborKernel


## Non-Linear filters:

# Median Filter
med = cv2.medianBlur(img, 15)
imshow_pair(med, img)


## Math morphology
m_elem = np.ones((7, 7))
eroded = cv2.erode(img, m_elem)    # -
dilated = cv2.dilate(img, m_elem)  # +
imshow_pair(dilated, eroded)

# close: erode(dilate(img))
cv2_close = lambda x, sz: \
    cv2.morphologyEx(x, cv2.MORPH_CLOSE,
                     np.ones((sz, sz)))

# open: dilate(erode(img))
cv2_open = lambda x, sz: \
    cv2.morphologyEx(x, cv2.MORPH_OPEN,
                     np.ones((sz, sz)))

opened = cv2_open(img, 10)
closed = cv2_close(img, 10)
imshow_pair(opened, closed)

# gradient: dilate - erode
imshow_gray(dilated - eroded)

# Corner detector: open - close
# Background normalization:
# a = open(close(img))
# a - open(close(a))


# Template matching, correlation
patch = img[230:365, 197:330].copy()
imshow_gray(patch)

match_map = cv2.matchTemplate(img, patch,
                              cv2.TM_CCOEFF_NORMED)
print(match_map.min(), match_map.max())
imshow_gray(match_map)


# Canny edge detector
smoothed = cv2.GaussianBlur(img, (0, 0),
                            sigmaX=3, sigmaY=3)
canny = cv2.Canny(smoothed, 75, 10)
imshow_pair(img, canny)
imshow_gray(canny)
