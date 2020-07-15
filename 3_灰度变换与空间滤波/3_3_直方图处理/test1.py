import hist
import cv2
import numpy as np

src_path = './pic/chapter03/p13.tif'
src = cv2.imread(src_path, 0)

src_path1 = './pic/chapter03/p9.tif'
src1 = cv2.imread(src_path1, 0)
m_hist = hist.arrayToHist(src1, 256)
hist.drawHist(m_hist)

dst = hist.hist_Match(src, m_hist)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
