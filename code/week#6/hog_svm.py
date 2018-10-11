from __future__ import print_function
from miptcv_utils import *
from os.path import join

rnd = lambda x: np.round(x).astype(int)

hog = cv2.HOGDescriptor()
detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
print(detector.shape)

hog.setSVMDetector(detector)

hog.getDescriptorSize()
print(hog.blockSize)
print(hog.blockStride)
print(hog.nbins)
print(hog.cellSize)
print(hog.winSize)


# INRIA pedestrian dataset
# http://pascal.inrialpes.fr/data/human/

root = ''
# img = cv2.imread(join(root, 'crop001001.png'))
# img = cv2.imread(join(root, 'crop001008.png'))
img = cv2.imread(join(root, 'crop001009.png'))
img = cv2.imread(join(root, 'crop001037.png'))


# detections = hog.detectMultiScale(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imshow(img)

min_scale, max_scale, stp_scale = 0.2, 1.0, 0.92
scales = [1.0]
while scales[-1] > min_scale:
    scales.append(scales[-1] * stp_scale)
scales = np.asarray(scales)

all_detections = []
for fx in scales:
    print('fx = %0.2f' % fx)

    y_scales = scales[(scales < fx / 0.8) & (scales > fx * 0.8)]

    for fy in y_scales:
        scaled = cv2.resize(img, None, fx=fx, fy=fy)
        detections, weights = hog.detect(scaled)

        if len(detections) == 0:
            continue

        for d, w in zip(detections, weights):
            if w > 0.1:
                pt1 = rnd(d[0] / fx), rnd(d[1] / fy)
                pt2 = rnd((d[0] + 64) / fx), rnd((d[1] + 128) / fy)
                all_detections.append((pt1, pt2))

show = img.copy()
for pt1, pt2 in all_detections:
    cv2.rectangle(show, pt1, pt2, (255,0,0), 2)
imshow(show)
