from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'week#1/tst_1.png'
img = cv2.imread(path)
assert img is not None

print('original shape:', img.shape)
print('type: %s' % type(img))

scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
print('scaled shape:', scaled.shape)


gray = scaled.sum(axis=2) / 3
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(scaled)
ax2.imshow(gray, cmap='gray')
f.show()


scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
plt.imshow(scaled)
# plt.show()

cv2.imshow('scaled', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Grey World
w, h, _ = scaled.shape

# tmp = np.sum(scaled, axis=0)
ch_ave = np.sum(np.sum(scaled, axis=0), axis=0).astype(float)
ch_ave /= w * h
print(ch_ave)
ch_ave /= ch_ave.mean()
grw = np.divide(scaled, ch_ave)

grw_ave = np.sum(np.sum(grw, axis=0), axis=0).astype(float)
grw_ave /= w * h

grw[grw > 255.0] = 255.0
grw = grw.astype(np.uint8)

# hist, bins = np.histogram(gray.ravel(), 256, [0, 256])

eq = scaled.copy()
for i in range(3):
    eq[:,:, i] = cv2.equalizeHist(scaled[:,:,i])

f, axes = plt.subplots(2, 3)
for ch in range(3):
    for i, img in enumerate([scaled, eq]):
        h = cv2.calcHist([img], channels=[ch], mask=None,
                         histSize=[256], ranges=[0, 256]).flatten()
        axes[i][ch].plot(range(256), np.cumsum(h))
f.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(scaled)
ax2.imshow(grw)
ax3.imshow(eq)

f.show()
