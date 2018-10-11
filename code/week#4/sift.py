from __future__ import print_function
from miptcv_utils import *


img1 = cv2.imread('week#5/box.png', 0)
img2 = cv2.imread('week#5/box_in_scene.png', 0)
imshow_pair(img1, img2)

sift = cv2.SIFT()

points1, descriptors1 = sift.detectAndCompute(img1, None)
points2, descriptors2 = sift.detectAndCompute(img2, None)

img1_show = cv2.drawKeypoints(img1, points1)
imshow_pair(img1, img1_show)

img2_show = cv2.drawKeypoints(img2, points2)
imshow(img2_show)
