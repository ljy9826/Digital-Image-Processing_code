import numpy as np
import matplotlib.pyplot as plt


def fft_distances(m, n):
    """
    计算m,n矩阵每一点距离中心的距离
    Args:
        m : 长
        n : 宽
    Returns:
        距离数组
    """

    M, N = np.meshgrid(np.arange(-n // 2, n // 2), np.arange(-m // 2, m // 2))
    D = np.sqrt(M**2 + N**2)
    return D


def IHPF(input_image, D0):
    """
    理想低通滤波器
    Args:
        input_image : 原图像
        D0 : 截止频率
    Returns:
        输出图像
    """
    input_image_cp = np.copy(input_image)

    # 傅里叶变换
    dft = np.fft.fft2(input_image_cp)
    dtf_shift = np.fft.fftshift(dft)
    # ifimg = np.log(np.abs(dtf_shift))

    w, h = input_image_cp.shape
    mask = np.zeros((w, h), np.uint8)
    D = fft_distances(w, h)
    for i in range(w):
        for j in range(h):
            if D[i][j] > D0:
                mask[i][j] = 1
    fshift = dtf_shift * mask

    # ofimg = np.log(np.abs(fshift))

    # 傅里叶反变换
    f_ishift = np.fft.ifftshift(fshift)
    output_image = np.fft.ifft2(f_ishift)
    output_image = np.abs(output_image)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(131), plt.imshow(ifimg, cmap='gray'), plt.title('input image')
    # plt.subplot(132), plt.imshow(ofimg, cmap='gray'), plt.title('output image')
    # plt.subplot(133), plt.imshow(mask, cmap='gray'), plt.title('mask image')
    # plt.show()

    return output_image.astype('uint8')


def BHPF(input_image, D0, N):
    """
    布特沃斯低通滤波器
    Args:
        input_image : 原图像
        D0 : 截止频率
        N : 阶数
    Returns:
        输出图像
    """
    input_image_cp = np.copy(input_image)

    # 傅里叶变换
    dft = np.fft.fft2(input_image_cp)
    dtf_shift = np.fft.fftshift(dft)
    # ifimg = np.log(np.abs(dtf_shift))

    w, h = input_image_cp.shape
    cx, cy = int(w / 2), int(h / 2)
    mask = np.zeros((w, h), np.uint8)
    D = fft_distances(w, h)
    mask = 1 / (1 + np.power(D0 / D, 2 * N))
    fshift = dtf_shift * mask

    # ofimg = np.log(np.abs(fshift))

    # 傅里叶反变换
    f_ishift = np.fft.ifftshift(fshift)
    output_image = np.fft.ifft2(f_ishift)
    output_image = np.abs(output_image)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(131), plt.imshow(ifimg, cmap='gray'), plt.title('input image')
    # plt.subplot(132), plt.imshow(ofimg, cmap='gray'), plt.title('output image')
    # plt.subplot(133), plt.imshow(mask, cmap='gray'), plt.title('mask image')
    # plt.show()

    return output_image.astype('uint8')


def GHPF(input_image, D0):
    """
    高斯低通滤波器
    Args:
        input_image : 原图像
        D0 : 截止频率
    Returns:
        输出图像
    """
    input_image_cp = np.copy(input_image)

    # 傅里叶变换
    dft = np.fft.fft2(input_image_cp)
    dtf_shift = np.fft.fftshift(dft)
    # ifimg = np.log(np.abs(dtf_shift))

    w, h = input_image_cp.shape
    cx, cy = int(w / 2), int(h / 2)
    mask = np.zeros((w, h), np.uint8)
    D = fft_distances(w, h)
    mask = 1 - np.exp(-(D * D) / (2 * D0 * D0))
    fshift = dtf_shift * mask

    # ofimg = np.log(np.abs(fshift))

    # 傅里叶反变换
    f_ishift = np.fft.ifftshift(fshift)
    output_image = np.fft.ifft2(f_ishift)
    output_image = np.abs(output_image)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(131), plt.imshow(ifimg, cmap='gray'), plt.title('input image')
    # plt.subplot(132), plt.imshow(ofimg, cmap='gray'), plt.title('output image')
    # plt.subplot(133), plt.imshow(mask, cmap='gray'), plt.title('mask image')
    # plt.show()

    return output_image.astype('uint8')


def homomorphic_filter(input_image, D0, rl, rh, c):
    """
    同态滤波器
    Args:
        input_image : 原图像
        D0 : 截止频率
        rl : 低频分量参数
        rh : 高频分量参数
        c : 系数
    Returns:
        输出图像
    """

    input_image_cp = np.copy(input_image)

    # 傅里叶变换
    dft = np.fft.fft2(input_image_cp)
    dtf_shift = np.fft.fftshift(dft)
    # ifimg = np.log(np.abs(dtf_shift))

    w, h = input_image_cp.shape
    mask = np.zeros((w, h), np.uint8)
    D = fft_distances(w, h)
    mask = (rh - rl) * (1 - np.exp(-c * (D**2 / D0**2))) + rl
    fshift = dtf_shift * mask

    # ofimg = np.log(np.abs(fshift))

    # 傅里叶反变换
    f_ishift = np.fft.ifftshift(fshift)
    output_image = np.fft.ifft2(f_ishift)
    output_image = np.abs(output_image)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(131), plt.imshow(ifimg, cmap='gray'), plt.title('input image')
    # plt.subplot(132), plt.imshow(ofimg, cmap='gray'), plt.title('output image')
    # plt.subplot(133), plt.imshow(mask, cmap='gray'), plt.title('mask image')
    # plt.show()

    return output_image.astype('uint8')