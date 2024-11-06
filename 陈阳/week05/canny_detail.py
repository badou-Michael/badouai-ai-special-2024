import numpy as np
import matplotlib.pyplot as plt
import math


# 1.高斯滤波
def gaussian_filter(img, sigma, dimensions):
    # sigma是高斯核参数，dimensions是尺寸，构造一个高斯核数组
    gaussian_filter_kernel = np.zeros([dimensions, dimensions])
    tmp = [i - dimensions // 2 for i in range(dimensions)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dimensions):
        for j in range(dimensions):
            gaussian_filter_kernel[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaussian_filter_kernel = gaussian_filter_kernel / gaussian_filter_kernel.sum()  # 求均值
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 得到的是浮点数
    tmp = dimensions // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), "constant")  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dimensions, j:j + dimensions] * gaussian_filter_kernel)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis("off")
    return img_new


# 2.求x方向和y方向的梯度
def gradient_calculation(img, img_new):  # 求梯度
    # 以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kennel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kennel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)
    dx, dy = img.shape
    img_tidu_y = np.zeros(img.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), "constant")  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kennel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kennel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] == 0.00000001
    tangent = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap="gray")
    plt.axis("off")
    return img_tidu, tangent


# 3.非极大值抑制
def nms(img, img_tidu, tangent):
    img_yizhi = np.zeros(img_tidu.shape)
    dx, dy = img.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 标记是否需要抹除
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8领域矩阵
            # 使用线性插值法判断是否抑制
            if tangent[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / tangent[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / tangent[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tangent[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / tangent[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / tangent[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tangent[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) / tangent[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) / tangent[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tangent[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) / tangent[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) / tangent[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap="gray")
    plt.axis("off")
    return img_yizhi


# 双阈值检测，连接边缘，遍历所有一定是边缘的点，查看8领域是否存在有可能是边的点，存进栈
def dual_threshold_detection(img_tidu, img_yizhi):
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 这里要取，这一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp1, temp2 = zhan.pop()  # 出栈
        a = img_yizhi[temp1 - 1:temp1 + 2, temp2 - 1:temp2 + 2]
        if high_boundary > a[0, 0] > lower_boundary:
            img_yizhi[temp1 - 1, temp2 - 1] = 255
            zhan.append([temp1 - 1, temp2 - 1])
        if high_boundary > a[0, 1] > lower_boundary:
            img_yizhi[temp1 - 1, temp2] = 255
            zhan.append([temp1 - 1, temp2])
        if high_boundary > a[0, 2] > lower_boundary:
            img_yizhi[temp1 - 1, temp2 + 1] = 255
            zhan.append([temp1 - 1, temp2 + 1])
        if high_boundary > a[1, 0] > lower_boundary:
            img_yizhi[temp1, temp2 - 1] = 255
            zhan.append([temp1, temp2 - 1])
        if high_boundary > a[1, 2] > lower_boundary:
            img_yizhi[temp1, temp2 + 1] = 255
            zhan.append([temp1, temp2 + 1])
        if high_boundary > a[2, 0] > lower_boundary:
            img_yizhi[temp1 + 1, temp2 - 1] = 255
            zhan.append([temp1 + 1, temp2 - 1])
        if high_boundary > a[2, 1] > lower_boundary:
            img_yizhi[temp1 + 1, temp2] = 255
            zhan.append([temp1 + 1, temp2])
        if high_boundary > a[2, 2] > lower_boundary:
            img_yizhi[temp1 + 1, temp2 + 1] = 255
            zhan.append([temp1 + 1, temp2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    pic_path = "../week02/lenna.png"
    img = plt.imread(pic_path)
    print("Image", img)
    if (pic_path[-4:]) == ".png":
        img = img * 255
    img = img.mean(axis=-1)  # 取均值的方法实现灰度化
    img_new = gaussian_filter(img, 0.5, 5)
    img_tidu, tangent = gradient_calculation(img,img_new)
    img_yizhi = nms(img, img_tidu, tangent)
    dual_threshold_detection(img_tidu, img_yizhi)
