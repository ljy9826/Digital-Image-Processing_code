import cv2
import hpfilter

src_path = './pic/chapter04/p1.tif'
src = cv2.imread(src_path, 0)

dst = hpfilter.IHPF(src, 360)  #理想高通滤波
# dst = hpfilter.BHPF(src, 10, 2)  #布特沃斯高通滤波
# dst = hpfilter.GHPF(src, 10)  #高斯高通滤波
# dst = hpfilter.homomorphic_filter(src, 80, 0.25, 2, 1)  #同态滤波
cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
