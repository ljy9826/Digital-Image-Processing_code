import basic_grayscale_transformation
import cv2
import numpy as np

src_path = './pic/chapter03/p8.tif'
src = cv2.imread(src_path, 0)

# dst = basic_grayscale_transformation.image_reverse(src)  #图像反转p1
# dst = basic_grayscale_transformation.logarithmic_transformation(src, 1)  #对数变换p2
# dst = basic_grayscale_transformation.gamma_transformation(src, 1, 0.6)  #伽马变换p3p4p5
# dst = basic_grayscale_transformation.contrast_stretch(src)  #对比度拉伸p6
# dst = basic_grayscale_transformation.grayscale_layer(src, 150, 200, 1)  #灰度级分层p7
dst1 = basic_grayscale_transformation.extract_bit_layer(src, 1)  #提取比特层p8
dst2 = basic_grayscale_transformation.extract_bit_layer(src, 2)
dst3 = basic_grayscale_transformation.extract_bit_layer(src, 3)
dst4 = basic_grayscale_transformation.extract_bit_layer(src, 4)
dst5 = basic_grayscale_transformation.extract_bit_layer(src, 5)
dst6 = basic_grayscale_transformation.extract_bit_layer(src, 6)
dst7 = basic_grayscale_transformation.extract_bit_layer(src, 7)
dst8 = basic_grayscale_transformation.extract_bit_layer(src, 8)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.imshow('dst5', dst5)
cv2.imshow('dst6', dst6)
cv2.imshow('dst7', dst7)
cv2.imshow('dst8', dst8)

cv2.waitKey(0)
