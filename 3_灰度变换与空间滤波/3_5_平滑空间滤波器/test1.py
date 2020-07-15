import smooth_space_filter
import cv2

src_path = './pic/chapter03/p16.tif'
src = cv2.imread(src_path, 0)

dst1 = smooth_space_filter.means_filter(src, 3)
dst2 = smooth_space_filter.median_filter(src, 3)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey(0)
