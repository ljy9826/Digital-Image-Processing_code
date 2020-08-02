import region_grow
import cv2
import numpy as np


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + 'Point' + '(' + str(x) + ', ' + str(y) + ')',
              src[y, x])
        # clicks.append((y, x))


src_path = './pic/chapter10/p3.tif'
src = cv2.imread(src_path, 0)
cv2.imshow('src', src)
cv2.setMouseCallback(
    'src',
    on_mouse,
    0,
)

seeds = [region_grow.Point(296, 255)]

dst = region_grow.regionGrow(src, seeds, 5)

cv2.imshow('dst', dst)

cv2.waitKey(0)
