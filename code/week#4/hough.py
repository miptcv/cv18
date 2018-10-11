from __future__ import print_function
from miptcv_utils import *
from skimage.transform import (
    hough_line, hough_line_peaks)


path = 'week#4/paper.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (612, 816))

# Canny edge detector
smoothed = cv2.GaussianBlur(
    img, (0, 0), sigmaX=3, sigmaY=3)


canny = cv2.Canny(smoothed, 75, 10)
width, height = canny.shape
imshow_pair(smoothed, canny)


h, theta, d = hough_line(canny)
h_wd, h_ht = h.shape
print(h.shape)
h_show = np.log(1 + h).T
imshow(h_show)


accum, angles, dists = \
    hough_line_peaks(h, theta, d)

print(accum)
print(angles)
print(dists)

f, ax = plt.subplots(1)
ax.imshow(h_show, cmap='gray')
ax.axis((0, h_wd, h_ht, 0))

for angle, dist in zip(*(angles, dists)):
    angle_idx = np.searchsorted(theta, angle)
    dist_idx = np.searchsorted(d, dist)
    ax.plot(dist_idx, angle_idx, 'ro')


f, ax = plt.subplots(1)
ax.imshow(canny, cmap='gray')
ax.axis((0, height, width, 0))
ax.set_axis_off()

def get_y(x, dist, angle):
    return (dist - x * np.cos(angle)) / \
           np.sin(angle)

for angle, dist in zip(*(angles, dists)):
    y0 = get_y(0, dist, angle)
    y1 = get_y(width, dist, angle)
    ax.plot((0, width), (y0, y1), 'r-')


# cv2:
# lines = cv2.HoughLines(canny, rho=1,
#         theta=np.pi/180, threshold=100)
