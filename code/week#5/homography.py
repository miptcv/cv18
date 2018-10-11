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


USE_FLANN_MATCHER = False
if USE_FLANN_MATCHER:
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches
                    if m.distance < 0.7 * n.distance]
else:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good_matches = bf.match(descriptors1.astype(np.uint8),
                            descriptors2.astype(np.uint8))


matches_img = make_matches_image(
    img1, points1, img2, points2, good_matches)
imshow(matches_img)


points1_m = np.asarray([points1[x.queryIdx].pt
                        for x in good_matches])
points2_m = np.asarray([points2[x.trainIdx].pt
                        for x in good_matches])

H, is_inlier = cv2.findHomography(
    points1_m, points2_m, method=cv2.RANSAC)

is_inlier = is_inlier.ravel().tolist()
print(sum(is_inlier))

matches_img = make_matches_image(
    img1, points1, img2, points2,
    good_matches, is_inlier)
imshow(matches_img)
