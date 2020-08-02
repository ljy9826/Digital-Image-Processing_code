import edge_detection
import cv2
import numpy as np

src_path = './pic/chapter10/p1.tif'
src = cv2.imread(src_path, 0)

src = cv2.resize(src, (480, 640))

dst = edge_detection.canny(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
# cv2.imshow('dst1', dst1)
# print(src.shape,dst.shape,dst1.shape)
# cv2.imshow('dilate', dilate)

cv2.waitKey(0)
