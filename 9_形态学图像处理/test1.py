import morphological_processing
import cv2
import numpy as np

src_path = './pic/chapter09/p4.tif'
src = cv2.imread(src_path, 0)

# src = np.zeros((5, 5), dtype=np.uint8)
# src.fill(255)
# src[3][3] = 0

# dst = morphological_processing.Erosion(src, 11)
# dst = morphological_processing.Dilation(src, 3)
# dst = morphological_processing.Open_operation(src)
# dst1 = morphological_processing.Close_operation(dst)
dst = morphological_processing.boundary_extraction(src)
# kernel = np.ones((3, 3), np.uint8)
# dilate = cv2.dilate(src, kernel, iterations=1)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
# cv2.imshow('dst1', dst1)
# cv2.imshow('dilate', dilate)

cv2.waitKey(0)
