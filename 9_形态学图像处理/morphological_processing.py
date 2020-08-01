import numpy as np


def Erosion(input_image, d):
    """
    腐蚀
    Args:
        input_image : 原图像
        d : 结构元大小
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    d_d = int(d / 2)
    w, h = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(0, w):
        for j in range(0, h):
            for x in range(-d_d, d_d + 1):
                for y in range(-d_d, d_d + 1):
                    if ((i + x) >= 0 and (i + x) < w and (j + y) >= 0
                            and (j + y) < h):
                        output_image[i][j] = np.bitwise_and(
                            output_image[i][j], input_image_cp[i + x][j + y])

    return output_image


def Dilation(input_image, d):
    """
    膨胀
    Args:
        input_image : 原图像
        d : 结构元大小
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    d_d = int(d / 2)
    w, h = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(0, w):
        for j in range(0, h):
            for x in range(-d_d, d_d + 1):
                for y in range(-d_d, d_d + 1):
                    if ((i + x) >= 0 and (i + x) < w and (j + y) >= 0
                            and (j + y) < h):
                        output_image[i][j] = np.bitwise_or(
                            output_image[i][j], input_image_cp[i + x][j + y])

    return output_image


def Open_operation(input_image):
    """
    开操作
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    img1 = Erosion(input_image_cp, 3)
    output_image = Dilation(img1, 3)

    return output_image


def Close_operation(input_image):
    """
    闭操作
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    img1 = Dilation(input_image_cp, 3)
    output_image = Erosion(img1, 3)

    return output_image


def boundary_extraction(input_image):
    """
    边界提取
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    output_image = input_image_cp - Erosion(input_image_cp, 3)

    return output_image

