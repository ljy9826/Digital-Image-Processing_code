import numpy as np


def image_reverse(input_image):
    """
    图像反转
    Args:
        input_image : 原图像
    Returns:
        反转后的图像
    """

    input_image_cp = np.copy(input_image)
    output_image = 255 - input_image_cp

    return output_image.astype('uint8')


def logarithmic_transformation(input_image, c):
    """
    对数变换
    Args:
        input_image : 原图像
        c ([type]): 对数变换系数c
    Returns:
        对数变换后的图像
    """

    input_image_cp = np.copy(input_image)
    output_image = c * np.log(
        1 + input_image_cp.astype(int))  #np.log函数需要变换成int型变量
    output_image = output_image / (
        c * np.log(1 + 255)) * 255  #需要将结果重新标定为[0，L-1]的灰度级

    return output_image.astype('uint8')


def gamma_transformation(input_image, c, gamma):
    """
    伽马变换
    Args:
        input_image : 原图像
        c : 伽马变换系数
        gamma : 幂次
    Returns:
        伽马变换后的图像
    """

    input_image_cp = np.copy(input_image)
    output_image = c * np.power(input_image_cp.astype(int), gamma)
    output_image = output_image / (
        c * np.power(255, gamma)) * 255  #需要将结果重新标定为[0，L-1]的灰度级

    return output_image.astype('uint8')


def contrast_stretch(input_image):
    """
    对比度拉伸（此实现为阈值处理，阈值为均值）
    Args:
        input_image : 原图像
    Returns:
        对比度拉伸后的图像
    """

    input_image_cp = np.copy(input_image)
    pixels_value_mean = np.mean(input_image_cp)
    input_image_cp[np.where(input_image_cp <= pixels_value_mean)] = 0
    input_image_cp[np.where(input_image_cp > pixels_value_mean)] = 255
    output_image = input_image_cp

    return output_image.astype('uint8')


def grayscale_layer(input_image, spotlight_range_min, spotlight_range_max,
                    flag):
    """
    灰度级分层
    Args:
        input_image : 原图像
        spotlight_range_min : 突出的灰度级范围最小值
        spotlight_range_max : 突出的灰度级范围最大值
        flag : 分层方式（1,2）
    Returns:
        灰度级分层后的图像
    """

    input_image_cp = np.copy(input_image)
    if flag == 1:  #方式1：突出范围内灰度为255，并将范围外灰度变为0
        input_image_cp = np.where((input_image_cp >= spotlight_range_min) &
                                  (input_image_cp <= spotlight_range_max), 255,
                                  0)
    elif flag == 2:  #方式2：突出范围内灰度为255，范围外灰度不变
        input_image_cp[np.where((input_image_cp >= spotlight_range_min) &
                                (input_image_cp <= spotlight_range_max))] = 255
    else:
        print('please enter the number of means from 1 to 2')
        return
    output_image = input_image_cp

    return output_image.astype('uint8')


def extract_bit_layer(input_image, layer_num):
    """
    提取比特层
    Args:
        input_image : 原图像
        layer_num : 提取层
    Returns:
        提取到的比特层图像
    """

    input_image_cp = np.copy(input_image)
    if layer_num == 1:
        input_image_cp = np.where((input_image_cp >= 0) & (input_image_cp < 2),
                                  255, 0)
    elif layer_num == 2:
        input_image_cp = np.where((input_image_cp >= 2) & (input_image_cp < 4),
                                  255, 0)
    elif layer_num == 3:
        input_image_cp = np.where((input_image_cp >= 4) & (input_image_cp < 8),
                                  255, 0)
    elif layer_num == 4:
        input_image_cp = np.where(
            (input_image_cp >= 8) & (input_image_cp < 16), 255, 0)
    elif layer_num == 5:
        input_image_cp = np.where(
            (input_image_cp >= 16) & (input_image_cp < 32), 255, 0)
    elif layer_num == 6:
        input_image_cp = np.where(
            (input_image_cp >= 32) & (input_image_cp < 64), 255, 0)
    elif layer_num == 7:
        input_image_cp = np.where(
            (input_image_cp >= 64) & (input_image_cp < 128), 255, 0)
    elif layer_num == 8:
        input_image_cp = np.where(
            (input_image_cp >= 128) & (input_image_cp < 256), 255, 0)
    else:
        print("please enter the number of bit layers from 1 to 8")

    output_image = input_image_cp

    return output_image.astype('uint8')
