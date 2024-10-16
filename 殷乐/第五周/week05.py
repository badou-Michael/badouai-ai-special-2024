import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


# 参数---sigma:高斯标准差 dim：高斯核尺寸
# 返回---高斯核
def gaussian_kernel(sigma, dim):
    gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    # G(x,y) = (1/2*pi*sigma**2)*e**[-(x**2+y**2)/2*sigma**2] = g1*e**[(x**2+y**2)/g2]
    # g1 = 1/2*pi*sigma**2
    g1 = 1 / (2 * math.pi * sigma ** 2)
    # g2 = -1/2*sigma**2
    g2 = -1 / 2 * sigma ** 2
    # 计算高斯核
    for x in range(dim):
        for y in range(dim):
            gaussian_filter[x, y] = g1 * math.exp(g2 * (tmp[x] ** 2 + tmp[y] ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    return gaussian_filter


# 参数---gaussian_filter：高斯核 img_gray：原图像  dim：高斯核尺寸
# 返回---高斯平滑后的图像
def gaussian_filter_img(gaussian_filter, img_gray, dim):
    # 填充大小
    pad_num = dim // 2
    # 边缘填充,'constant' 填充方式
    img_pad = np.pad(img_gray, ((pad_num, pad_num), (pad_num, pad_num)), 'constant')
    # 平滑后的图像
    img_filter = np.zeros(img_gray.shape)
    for x in range(img_gray.shape[0]):
        for y in range(img_gray.shape[1]):
            img_filter[x, y] = np.sum((img_pad[x:x + dim, y:y + dim]) * gaussian_filter)
    return img_filter


# 参数---原图像
# 返回---梯度图像和提取边缘后的图像
def sobel(img_filter):
    # 构建sobel算子
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 存储梯度图像
    img_gradient_x = np.zeros(img_filter.shape)
    img_gradient_y = np.zeros(img_filter.shape)
    img_gradient = np.zeros(img_filter.shape)
    # 使用sobel算子进行平滑(实现边缘检测)
    img_pad = np.pad(img_filter, ((1, 1), (1, 1)), 'constant')
    for x in range(img_filter.shape[0]):
        for y in range(img_filter.shape[1]):
            img_gradient_x[x, y] = np.sum((img_pad[x:x + 3, y:y + 3]) * sobel_kernel_x)
            img_gradient_y[x, y] = np.sum((img_pad[x:x + 3, y:y + 3]) * sobel_kernel_y)
            img_gradient[x, y] = np.sqrt(img_gradient_x[x, y] ** 2 + img_gradient_y[x, y] ** 2)
    return img_gradient, img_gradient_y, img_gradient


# 参数---theta：最大梯度夹角  img_gradient：原图
# 返回---非极大值抑制后的图
def non_maximum_suppression(theta, img_gradient):
    # 确定像素点局部最大值，非极大值点的灰度置为0；
    # 1) 将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。
    # 2) 如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则
    # 该像素点将被抑制（灰度值置为0）
    img_NMS = np.zeros(img_gradient.shape)
    # 遍历像素点：出去边缘像素
    for x in range(1, img_gradient.shape[0] - 1):
        for y in range(1, img_gradient.shape[1] - 1):
            # 判断是否保持原值的标记，false则设置为0
            flag = True
            # 像素点邻域
            temp = img_gradient[x - 1:x + 2, y - 1:y + 2]
            # 根据梯度不同使用不通的相邻点进行计算，使用双线性插值法
            if theta[x, y] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / theta[x, y] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / theta[x, y] + temp[2, 1]
                if not (img_gradient[x, y] > num_1 and img_gradient[x, y] > num_2):
                    flag = False
            elif theta[x, y] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / theta[x, y] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / theta[x, y] + temp[2, 1]
                if not (img_gradient[x, y] > num_1 and img_gradient[x, y] > num_2):
                    flag = False
            elif theta[x, y] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * theta[x, y] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * theta[x, y] + temp[1, 0]
                if not (img_gradient[x, y] > num_1 and img_gradient[x, y] > num_2):
                    flag = False
            elif theta[x, y] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * theta[x, y] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * theta[x, y] + temp[1, 2]
                if not (img_gradient[x, y] > num_1 and img_gradient[x, y] > num_2):
                    flag = False
            if flag:
                img_NMS[x, y] = img_gradient[x, y]
    return img_NMS


# 参数---high_boundary：高阈值  lower_boundary：低阈值  img_NMS：原图
# 返回---非双阈值检测和边缘连接后的图
def threshold_detection_and_edge_connection(high_boundary, lower_boundary, img_NMS):
    # 如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；
    # 如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；
    # 如果边缘像素的梯度值小于低阈值，则会被抑制。
    # 定义栈，用于存储强边缘像素，便于后续连接边缘时进行判断
    strong_edge = []
    # 遍历像素点：除去边缘像素
    for i in range(1, img_NMS.shape[0] - 1):
        for j in range(1, img_NMS.shape[1] - 1):
            # 如果大于高阈值，强边缘，设置为255
            if img_NMS[i, j] >= high_boundary:
                img_NMS[i, j] = 255
                strong_edge.append([i, j])
            # 如果小于低阈值，抑制，设为0
            elif img_NMS[i, j] <= lower_boundary:
                img_NMS[i, j] = 0
    while not len(strong_edge) == 0:
        # 获取强边缘点
        temp_1, temp_2 = strong_edge.pop()
        # 获取强边缘点的邻域
        a = img_NMS[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for x in range(a.shape[0]):
            for y in range(a.shape[1]):
                # 如果强边缘的相邻点是若边缘点，设置为强边缘点
                if a[x, y] == 255:
                    continue
                if (a[x, y] < high_boundary) and (a[x, y] > lower_boundary):
                    img_NMS[temp_1 - 1, temp_2 - 1] = 255
                    strong_edge.append([temp_1 - 1, temp_2 - 1])
    # 剩余弱边缘点置为0
    for i in range(img_NMS.shape[0]):
        for j in range(img_NMS.shape[1]):
            if img_NMS[i, j] != 0 and img_NMS[i, j] != 255:
                img_NMS[i, j] = 0
    return img_NMS


def canny_detail():
    img = cv2.imread("lenna.png")
    # 1. 对图像进行灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(231)
    plt.imshow(img_gray, cmap='gray')
    # 2. 对图像进行高斯滤波：
    # 根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
    # 可以有效滤去理想图像中叠加的高频噪声。
    # 2.1 高斯标准差和高斯核尺寸
    sigma = 0.5
    dim = 5
    # 2.2 构建高斯核
    gaussian_filter = gaussian_kernel(sigma, dim)
    # 平滑后的图像
    img_filter = gaussian_filter_img(gaussian_filter, img_gray, dim)
    plt.subplot(232)
    plt.imshow(img_filter, cmap='gray')

    # 3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
    # 存储梯度图像
    img_gradient_x, img_gradient_y, img_gradient = sobel(img_filter)
    plt.subplot(233)
    plt.imshow(img_gradient.astype(np.uint8), cmap='gray')
    # 4 对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点
    # 所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
    img_gradient_x[img_gradient_x == 0] = 0.00000001
    theta = img_gradient_y / img_gradient_x
    img_NMS = non_maximum_suppression(theta, img_gradient)
    plt.subplot(234)
    plt.imshow(img_NMS.astype(np.uint8), cmap='gray')
    # 5 用双阈值算法检测和连接边缘
    # 设置高低阈值
    lower_boundary = img_gradient.mean() * 0.5
    high_boundary = lower_boundary * 3
    img_NMS = threshold_detection_and_edge_connection(high_boundary, lower_boundary, img_NMS)
    plt.subplot(235)
    plt.imshow(img_NMS, cmap='gray')
    plt.show()


def canny():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("canny", cv2.Canny(gray, gray.mean()*0.5, gray.mean()*1.5))
    cv2.waitKey()


canny_detail()
# canny()
