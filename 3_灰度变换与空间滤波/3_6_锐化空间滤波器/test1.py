import sharpen_space_filter
import cv2

src_path = './pic/chapter03/p18.tif'
src = cv2.imread(src_path, 0)

# dst = sharpen_space_filter.laplace_sharpen(src, -1)

dst = sharpen_space_filter.sharpen_gradient(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
