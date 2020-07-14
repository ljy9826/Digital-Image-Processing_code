import hist
import cv2
import numpy as np

src_path = './pic/chapter03/p9.tif'
src = cv2.imread(src_path, 0)

# m_hist = hist.arrayToHist(src, 256)
# hist.drawHist(m_hist)

dst = hist.hist_equalization(src)

m_hist = hist.arrayToHist(dst, 256)
hist.drawHist(m_hist)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(0)
