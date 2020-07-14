import numpy as np
from matplotlib import pyplot as plt


#将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(input_image, nums):
    """
    将灰度数组映射为直方图字典
    Args:
        input_image : 原图像
        nums : 灰度级

    Returns:
        直方图字典
    """

    input_image_cp = np.copy(input_image)
    w, h = input_image_cp.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            hist[input_image_cp[i][j]] += 1
    n = w * h
    for k in range(nums):
        hist[k] = float(hist[k]) / n

    return hist


def drawHist(hist):
    """[summary]
    绘制直方图
    Args:
        hist : 直方图字典
    """

    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  #x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    plt.bar(tuple(keys), tuple(values))  #绘制直方图
    plt.show()


def hist_equalization(input_image):
    """
    直方图均衡
    Args:
        input_image : 原图像
    Returns:
        直方图均衡后的图像
    """

    input_image_cp = np.copy(input_image)
    output_image = np.copy(input_image)

    input_image_grayscale_P = arrayToHist(input_image_cp, 256)

    t = 0
    for k in range(256):
        t += input_image_grayscale_P[k]
        output_image[np.where(input_image_cp == k)] = 255 * t

    return output_image

