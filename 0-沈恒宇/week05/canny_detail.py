"""
不调用canny接口，手动实现canny算法
一、灰度化
二、高斯平滑
三、sobel滤波
四、非极大值抑制
五、双阈值检测

对图像进行滤波操作的步骤：
1.（计算出）滤波核  2.创建相同尺寸的空白图像 3.边缘填充(dim//2) 4.遍历原图像，通过滤波加权求和 5.（还原为unit8）得到处理后图像
"""

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


# 1、图像灰度化
def fun_gray_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 2、高斯平滑
def fun_gaussian_image(gray_image):
    # 设置高斯函数的标准差，和高斯核的尺寸
    sigma = 0.5  # 标准差，可调
    dim = 5  # 高斯核尺寸，可调，这里是5x5
    # 创建高斯核
    # np.zeros()用于创建一个给定形状和数据类型的新数组，并将数组中的所有元素初始化为0。
    gaussian_filter = np.zeros([dim, dim])  # [dim,dim]指定了新创建的数组的行数和列数
    '''
    下面这段代码通常用于生成一个中心对称的序列，例如在图像处理或机器学习中，可能需要创建一个以原点为中心的网格或坐标系。
    //代表整除，确保dim是一个正整数，否则range(dim)将不会生成任何元素，tmp将是一个空列表。
    range(dim)：range函数生成一个从0到dim-1的整数序列。
    i-dim//2：对于range(dim)中的每个整数i，减去dim的一半（使用//进行整数除法）。
    例如：dim=5，则tmp=[-2,-1,0,1,2]
    '''
    tmp = [i-dim//2 for i in range(dim)]

    '''
    计算高斯核
    这里参考，高斯函数的表达式，看笔记里面有公式
    '''
    n1 = 1/(2*math.pi*sigma**2)  # 1/(2*π*σ²)
    n2 = -1/(2*sigma**2)  # -1/(2*σ²)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    '''
    归一化，使得所有元素的和为1
    归一化后的高斯核可以确保在图像与高斯核进行卷积操作时，图像的亮度不会因为核的加权而改变。
    换句话说，归一化后的高斯核不会改变图像的整体亮度，只是对图像进行平滑或模糊处理。
    '''
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    dx, dy = gray_image.shape
    gaussian_image = np.zeros(gray_image.shape)  # 储存平滑之后的图像，注意：np.zeros函数返回的是浮点型数据，创建的是一个数组
    '''
    对灰度图像进行边缘填补，以避免在滤波过程中出现索引越界的情况。
    np.pad函数在图像的上下左右各填补tmp行或列
    这里补充 dim对2取整 个,一般滤波都加dim的一半取整很合理
    (num1, num2),(num3, num4)表示在图像的上下左右各填补num1行、num2行、num3列和num4列
    填补方式为'constant'，即用常数填充，默认是用0填充
    '''
    tmp = dim//2
    gray_image_pad = np.pad(gray_image, ((tmp, tmp),(tmp, tmp)), 'constant')  # 边缘填补
    '''
    高斯滤波处理图像
    gray_image_pad[i:i+dim, j:j+dim]:提取的是以 (i, j) 为左上角，大小为 dim x dim 的子图像块。
    np.sum()：加权求和
    '''
    for i in range(dx):
        for j in range(dy):
            gaussian_image[i,j] = np.sum(gray_image_pad[i:i+dim, j:j+dim]*gaussian_filter)
    '''
    np.uint8 表示8位无符号整型，范围是0到255。
    在图像处理中，像素值通常用8位无符号整型表示，因为每个像素的颜色值（如RGB值）都可以用0到255之间的整数表示。
    '''
    gaussian_image = gaussian_image.astype(np.uint8)  # 转化为图片类型
    return gaussian_image


# 3、sobel算子。利用sobel矩阵求梯度（检测图像中的水平、垂直和对角边缘）
def fun_sobel_kernel(gaussian_image):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = np.zeros(gaussian_image.shape)
    gradient_y = np.zeros(gaussian_image.shape)
    gradient_of_image = np.zeros(gaussian_image.shape)
    image_pad = np.pad(gaussian_image, ((1,1), (1,1)), 'constant')
    dx, dy = gaussian_image.shape
    for i in range(dx):
        for j in range(dy):
            gradient_x[i, j] = np.sum(image_pad[i:i+3, j:j+3]*sobel_kernel_x)
            gradient_y[i, j] = np.sum(image_pad[i:i+3, j:j+3]*sobel_kernel_y)
            gradient_of_image[i, j] = np.sqrt(gradient_x[i,j]**2 + gradient_y[i, j]**2)
    gradient_x[gradient_x == 0] = 0.0000001   # 梯度方向
    tan = gradient_y / gradient_x
    sobel_image = gradient_of_image.astype(np.uint8)
    return tan, gradient_of_image, sobel_image


# 4、非极大值抑制Non-Maximum Suppression
def fun_nms(tan_of_image, gradient_of_image, sobel_image):
    suppression_image = np.zeros(sobel_image.shape)
    dx, dy = suppression_image.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = gradient_of_image[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if tan_of_image[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / tan_of_image[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan_of_image[i, j] + temp[2, 1]
                if not (gradient_of_image[i, j] > num_1 and gradient_of_image[i, j] > num_2):
                    flag = False
            elif tan_of_image[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan_of_image[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan_of_image[i, j] + temp[2, 1]
                if not (gradient_of_image[i, j] > num_1 and gradient_of_image[i, j] > num_2):
                    flag = False
            elif tan_of_image[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan_of_image[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan_of_image[i, j] + temp[1, 0]
                if not (gradient_of_image[i, j] > num_1 and gradient_of_image[i, j] > num_2):
                    flag = False
            elif tan_of_image[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan_of_image[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan_of_image[i, j] + temp[1, 2]
                if not (gradient_of_image[i, j] > num_1 and gradient_of_image[i, j] > num_2):
                    flag = False
            if flag:
                suppression_image[i, j] = gradient_of_image[i, j]
    suppression_image = suppression_image.astype(np.uint8)
    return suppression_image


# 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
def fun_dtd(suppression_image):
    canny_image = suppression_image
    lower_boundary = canny_image.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, canny_image.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, canny_image.shape[1] - 1):
            if canny_image[i, j] >= high_boundary:  # 取，一定是边的点
                canny_image[i, j] = 255
                zhan.append([i, j])
            elif canny_image[i, j] <= lower_boundary:  # 舍
                canny_image[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = canny_image[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            canny_image[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            canny_image[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            canny_image[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            canny_image[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            canny_image[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            canny_image[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            canny_image[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            canny_image[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(canny_image.shape[0]):
        for j in range(canny_image.shape[1]):
            if canny_image[i, j] != 0 and canny_image[i, j] != 255:
                canny_image[i, j] = 0
    canny_image = canny_image.astype(np.uint8)
    return canny_image


if __name__ == '__main__':
    image_grb = cv2.imread('lenna.png')
    image_rgb = cv2.cvtColor(image_grb, cv2.COLOR_BGR2RGB)
    image_gray = fun_gray_image(image_rgb)
    image_gaussian = fun_gaussian_image(image_gray)
    image_tan, image_gradient, image_sobel = fun_sobel_kernel(image_gaussian)
    image_suppression = fun_nms(image_tan,image_gradient,image_sobel)
    image_canny = fun_dtd(image_suppression)

    plt.subplot(231), plt.imshow(image_rgb), plt.title('rgb_image')
    plt.subplot(232), plt.imshow(image_gray, cmap='gray'), plt.title('gray_image')
    plt.subplot(233), plt.imshow(image_gaussian, cmap='gray'), plt.title('gaussian_image')
    plt.subplot(234), plt.imshow(image_sobel, cmap='gray'), plt.title('sobel_image')
    plt.subplot(235), plt.imshow(image_suppression, cmap='gray'), plt.title('suppression_image')
    plt.subplot(236), plt.imshow(image_canny, cmap='gray'), plt.title('canny_image')

    plt.show()










