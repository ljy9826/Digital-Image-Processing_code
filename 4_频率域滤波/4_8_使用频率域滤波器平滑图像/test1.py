import cv2
import lpfilter

src_path = './pic/chapter04/p1.tif'
src = cv2.imread(src_path, 0)

# dst = lpfilter.ILPF(src, 10)  #理想低通滤波

# dst = lpfilter.BLPF(src, 160, 2)  #布特沃斯低通滤波

dst = lpfilter.GLPF(src, 160)  #高斯低通滤波

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
