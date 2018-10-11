import matplotlib.pyplot as plt
import numpy as np
import cv2

def imshow_ax(img, ax):
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)


def imshow(img):
    f, ax = plt.subplots(1)
    imshow_ax(img, ax)
    plt.xticks([]); plt.yticks([])
    f.show()


def imshow_pair(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2)

    imshow_ax(img1, ax1)
    imshow_ax(img2, ax2)
    plt.xticks([]); plt.yticks([])
    f.show()


def make_matches_image(img1, kp1, img2, kp2, matches, which_to_draw=None):
    if which_to_draw is None:
        which_to_draw = np.zeros(len(matches))

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    def draw_match(img1_idx, img2_idx, color):
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt


        cv2.circle(out, (int(x1), int(y1)), 4, color, 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, color, 1)
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2) + cols1, int(y2)), color, 1)

    for idx, mat in enumerate(matches):
        if which_to_draw[idx] == 0:
            draw_match(mat.queryIdx, mat.trainIdx, (255, 0, 0))

    for idx, mat in enumerate(matches):
        if which_to_draw[idx] == 1:
            draw_match(mat.queryIdx, mat.trainIdx, (0, 255, 0))

    return out
