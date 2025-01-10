import numpy as np
import cv2

# --------------------------------------------------------
#  最邻近插值
def nearest_interpolation(image, scale_factor):
    """
    :param image: 输入图像
    :param scale_factor: 缩放因子
    :return: 缩放后的图像
    """
    # 获取原始图像的尺寸
    height, width, channels = image.shape
    # 计算目标图像的尺寸
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 创建目标图像
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # 进行最邻近插值
    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的坐标
            src_x = int(j / scale_factor)
            src_y = int(i / scale_factor)
            # 将原始图像中对应的像素值赋给目标图像
            new_image[i, j] = image[src_y, src_x]

    return new_image


# 读取图像
image = cv2.imread('lenna.png')
# 设置缩放因子
scale_factor = 2.0
# 进行最邻近插值
resized_image = nearest_interpolation(image, scale_factor)

# 使用接口函数进行最邻近插值
height1, width1, channels1 = resized_image.shape
resized_image1 = cv2.resize(image, (width1, height1), interpolation=cv2.INTER_NEAREST)

# 显示图像
cv2.imshow('Original Image', image)
cv2.imshow('Nearest Interpolation', resized_image)
cv2.imshow('Nearest Interpolation cv2', resized_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------
#  双线性插值

def bilinear_interpolation(image, scale_factor):
    """
    双线性插值函数
    :param image: 输入图像
    :param scale_factor: 缩放因子
    :return: 缩放后的图像
    """
    # 获取原始图像的尺寸
    height, width, channels = image.shape
    # 计算目标图像的尺寸
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 创建目标图像
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # 进行双线性插值
    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的坐标
            src_x = j / scale_factor
            src_y = i / scale_factor

            # 找到原始图像中四个最近的像素点的坐标
            x1, y1 = int(src_x), int(src_y)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

            # 获取四个像素点的值
            Ia = image[y1, x1]
            Ib = image[y1, x2]
            Ic = image[y2, x1]
            Id = image[y2, x2]

            # 计算插值
            wa = (x2 - src_x) * (y2 - src_y)
            wb = (src_x - x1) * (y2 - src_y)
            wc = (x2 - src_x) * (src_y - y1)
            wd = (src_x - x1) * (src_y - y1)

            # 计算目标像素的值
            new_image[i, j] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return new_image

scale_factor = 2.0
# 进行双线性插值
resized_image_b = bilinear_interpolation(image, scale_factor)
# 显示图像
# 使用接口函数进行最邻近插值
height1, width1, channels1 = resized_image.shape
resized_image_b1 = cv2.resize(image, (width1, height1), interpolation=cv2.INTER_LINEAR)

# 显示图像
cv2.imshow('Original Image', image)
cv2.imshow('Bilinear Interpolation', resized_image_b)
cv2.imshow('Bilinear Interpolation cv2', resized_image_b1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------
# 直方图均衡化
import matplotlib.pyplot as plt

def equalize_histogram_opencv(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 进行直方图均衡化
    equalized_image = cv2.equalizeHist(image)

    # 直方图
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # 创建一个图形窗口
    plt.figure(figsize=(12, 6))

    # 显示原始图像
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    # 显示均衡化后的图像
    plt.subplot(2, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')

    # 显示原始图像的直方图
    plt.subplot(2, 2, 3)
    plt.title('Histogram of Original Image')
    plt.plot(hist_original)
    plt.xlim([0, 256])

    # 显示均衡化后图像的直方图
    plt.subplot(2, 2, 4)
    plt.title('Histogram of Equalized Image')
    plt.plot(hist_equalized)
    plt.xlim([0, 256])

    plt.show()

    return equalized_image

# 示例用法
equalized_image = equalize_histogram_opencv('lenna.png')
