import numpy as np


def means_filter(input_image, filter_size):
    """
    均值滤波
    Args:
        input_image : 原图像
        filter_size : 滤波器大小
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)
    filter_template = np.ones((filter_size, filter_size))
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num),
                            mode="constant",
                            constant_values=0)
    output_image = input_image_cp.copy()

    w, h = input_image_cp.shape
    for i in range(pad_num, w - pad_num):
        for j in range(pad_num, h - pad_num):
            output_image[i][j] = np.sum(
                filter_template *
                input_image_cp[i - pad_num:i + pad_num + 1,
                               j - pad_num:j + pad_num + 1]) / (filter_size**2)

    output_image = output_image[pad_num:w - pad_num, pad_num:h - pad_num]

    return output_image


def median_filter(input_image, filter_size):
    """
    中值滤波
    Args:
        input_image : 原图像
        filter_size : 滤波器大小
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num),
                            mode="constant",
                            constant_values=0)
    output_image = input_image_cp.copy()

    w, h = input_image_cp.shape
    for i in range(pad_num, w - pad_num):
        for j in range(pad_num, h - pad_num):
            output_image[i][j] = np.median(
                input_image_cp[i - pad_num:i + pad_num + 1,
                               j - pad_num:j + pad_num + 1])

    output_image = output_image[pad_num:w - pad_num, pad_num:h - pad_num]

    return output_image

