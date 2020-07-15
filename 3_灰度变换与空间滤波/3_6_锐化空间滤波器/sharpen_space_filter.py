import numpy as np
import scipy.signal

def laplace_sharpen(input_image, c):
    """
    拉普拉斯锐化
    Args:
        input_image : 原图像
        c : 锐化系数
    Returns:
        输出图像
    """

    input_image_cp = input_image.copy()
    laplace_filter = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])
    # laplace_filter = np.array([
    #     [1, 1, 1],
    #     [1, -8, 1],
    #     [1, 1, 1],
    # ])
    input_image_cp = np.pad(input_image_cp, (1, 1),
                            mode='constant',
                            constant_values=0)
    w, h = input_image_cp.shape
    output_image = np.copy(input_image_cp)

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            R = np.sum(laplace_filter *
                       input_image_cp[i - 1:i + 2, j - 1:j + 2])
            output_image[i, j] = input_image_cp[i, j] + c * R

    return output_image


def sharpen_gradient(input_image):
    """
    梯度锐化
    Args:
        input_image : 原图像
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)  # 输入图像的副本

    # #罗伯特交叉梯度算子
    # roberts_kernel = np.array([[-1, 0], [0, 1]])

    # sobel算子
    sobel_kernel_1 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ])
    sobel_kernel_2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])

    output_image = np.abs(
        scipy.signal.convolve2d(
            input_image_cp, sobel_kernel_1, mode='same')) + np.abs(
                scipy.signal.convolve2d(
                    input_image_cp, sobel_kernel_2, mode='same'))

    return output_image.astype('uint8')
