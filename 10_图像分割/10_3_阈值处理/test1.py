import threshold_processing
import cv2
import numpy as np

src_path = './pic/chapter10/p2.tif'
src = cv2.imread(src_path, 0)

dst = threshold_processing.basic_threshold(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
