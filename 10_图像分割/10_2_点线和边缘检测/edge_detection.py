import numpy as np
import math


def gaussian_smooth(input_image):
    """
    高斯滤波
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)

    # 生成高斯滤波器
    """
    常用尺寸为 5x5，σ=1.4 的高斯滤波器
    H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
    """
    sigma = 1.4
    gau_sum = 0
    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(
                (-1 / (2 * sigma * sigma)) *
                (np.square(i - 3) + np.square(j - 3))) / (2 * math.pi * sigma *
                                                          sigma)
            gau_sum = gau_sum + gaussian[i, j]

    # 归一化处理
    gaussian = gaussian / gau_sum

    # 高斯滤波
    w, h = input_image.shape
    output_image = np.zeros([w, h])

    for i in range(2, w - 2):
        for j in range(2, h - 2):
            output_image[i, j] = np.sum(
                gaussian * input_image_cp[i - 2:i + 3, j - 2:j + 3])

    return output_image


def cal_gradients(input_image):
    """
    计算梯度幅值
    Args:
        input_image : 原图像

    Returns:
        dx: x方向梯度值
        dy: y方向梯度值
        M: 梯度幅值
        theta: 梯度方向角
    """
    input_image_cp = np.copy(input_image).astype('float64')

    w, h = input_image_cp.shape
    dx = np.zeros([w, h])
    dy = np.zeros([w, h])
    M = np.zeros([w, h])
    theta = np.zeros([w, h])

    for i in range(w - 1):
        for j in range(h - 1):
            dx[i, j] = input_image_cp[i + 1, j] - input_image_cp[i, j]
            dy[i, j] = input_image_cp[i, j + 1] - input_image_cp[i, j]
            M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
            theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.000000001))

    return dx, dy, M, theta


def cal_NMS(M, dx, dy):
    """
    非极大值抑制
    Args:
        M: 梯度幅值
        dx : x方向梯度值
        dy : y方向梯度值
    Returns:
        非极大值抑制结果
    """

    d = np.copy(M)
    w, h = M.shape
    NMS = np.copy(d)
    # NMS[0, :] = NMS[w-1, :] = NMS[:, 0] = NMS[:, h-1] = 0

    for i in range(1, w - 1):
        for j in range(1, h - 1):

            # 如果当前梯度为0，该点就不是边缘点
            if M[i, j] == 0:
                NMS[i, j] = 0

            else:
                gradX = dx[i, j]  # 当前点 x 方向导数
                gradY = dy[i, j]  # 当前点 y 方向导数
                gradTemp = d[i, j]  # 当前梯度点

                # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)  # 权重
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1 g2
                    #    c
                    #    g4 g3
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #    g2 g1
                    #    c
                    # g3 g4
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果 x 方向梯度值比较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1
                    if gradX * gradY > 0:

                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    # g1
                    # g2 c g4
                    #      g3
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                # 利用 grad1-grad4 对梯度进行插值
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4

                # 当前像素的梯度是局部的最大值，可能是边缘点
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp

                else:
                    # 不可能是边缘点
                    NMS[i, j] = 0

    return NMS


def double_threshold(NMS):
    """
    双阈值选取
    Args:
        NMS : 非极大值抑制结果
    Returns:
        输出图像
    """
    w, h = NMS.shape
    output_image = np.zeros([w, h])
    # 定义高低阈值
    TL = 0.1 * np.max(NMS)
    TH = 0.3 * np.max(NMS)

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            # 双阈值选取
            if (NMS[i, j] < TL):
                output_image[i, j] = 0

            elif (NMS[i, j] > TH):
                output_image[i, j] = 1

        # 连接
            elif (NMS[i - 1, j - 1:j + 1] <
                  TH).any() or (NMS[i + 1, j - 1:j + 1].any() or
                                (NMS[i, [j - 1, j + 1]] < TH).any()):
                output_image[i, j] = 1

    return output_image


def canny(input_image):
    """
    坎尼边缘检测
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """
    input_image_cp = np.copy(input_image)
    smooth_img = gaussian_smooth(input_image_cp)
    dx, dy, M, theta = cal_gradients(smooth_img)
    NMS = cal_NMS(M, dx, dy)
    output_image = double_threshold(NMS)

    return output_image