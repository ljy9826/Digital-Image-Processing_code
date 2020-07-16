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

    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1
    ret = np.sqrt(u * u + v * v)

    return np.fft.fftshift(ret)


def ILPF(input_image, D0):
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
    # cx, cy = int(w / 2), int(h / 2)
    mask = np.zeros((w, h), np.uint8)
    D = fft_distances(w, h)
    for i in range(w):
        for j in range(h):
            # D = np.sqrt((i - cx)**2 + (j - cy)**2)
            if D[i][j] <= D0:
                mask[i][j] = 1
    # mask[cx - D0:cx + D0, cy - D0:cy + D0] = 1
    fshift = dtf_shift * mask

    # ofimg = np.log(np.abs(fshift))

    # 傅里叶反变换
    f_ishift = np.fft.ifftshift(fshift)
    output_image = np.fft.ifft2(f_ishift)
    output_image = np.abs(output_image)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(121), plt.imshow(ifimg, cmap='gray'), plt.title('input image')
    # plt.subplot(122), plt.imshow(ofimg, cmap='gray'), plt.title('output image')
    # plt.show()

    return output_image.astype('uint8')


def BLPF(input_image, D0, N):
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
    mask = 1 / (1 + np.power(D / D0, 2 * N))
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


def GLPF(input_image, D0):
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
    mask = np.exp(-(D * D) / (2 * D0 * D0))
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
