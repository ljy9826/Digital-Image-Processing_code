import numpy as np


def basic_threshold(input_image):
    """
    基本的全局阈值处理
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)
    T = np.mean(input_image_cp)
    done = False

    while ~done:
        g1 = input_image_cp[np.where(input_image_cp > T)]
        g2 = input_image_cp[np.where(input_image_cp <= T)]
        Tnext = 0.5 * (np.mean(g1) + np.mean(g2))
        done = abs(T - Tnext) < 0.5
        T = Tnext

    output_image = np.copy(input_image_cp)
    output_image[np.where(input_image_cp > T)] = 255
    output_image[np.where(input_image_cp <= T)] = 0

    return output_image
