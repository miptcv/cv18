from __future__ import print_function
from miptcv_utils import *

path = 'week#4/paper.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (612, 816))
height, width = img.shape
imshow(img)


smoothed = cv2.GaussianBlur(
    img, (0, 0), sigmaX=5, sigmaY=5)

harris_map = cv2.cornerHarris(
    smoothed, blockSize=10, ksize=1, k=0.04)
imshow_pair(smoothed, harris_map)


# 1. Harris
# 2. Non-maximum suppression
# 3. Reject harris_map which less than qualityLevel * harris_map.max()
# 4. Reject corners, which closer to each other more than minDistance
# 5. Return top maxCorners
points = cv2.goodFeaturesToTrack(
    smoothed, maxCorners=25, qualityLevel=0.01, minDistance=10,
    useHarrisDetector=True, blockSize=10, k=0.02)
print(points)


f, ax = plt.subplots(1)
ax.imshow(smoothed, cmap='gray')
ax.axis((0, width, height, 0))
ax.set_axis_off()
ax.plot(points[:,:,0], points[:,:,1], 'ro')
