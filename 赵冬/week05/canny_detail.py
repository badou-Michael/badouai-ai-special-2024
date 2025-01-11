import math

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 读取图像
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        img = img * 255

    # 灰度化
    img = img.mean(axis=-1)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img, cmap='gray')

    # 高斯滤波
    sigma = 0.5
    dim = 5
    Gaussian_filter = np.zeros((dim, dim))
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)

    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

    print(Gaussian_filter)
    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    padding = dim // 2
    img_pad = np.pad(img, ((padding, padding), (padding, padding)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)

    # 边缘提取
    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    img_grad_x = np.zeros(img.shape)
    img_grad_y = np.zeros(img.shape)
    img_grad = np.zeros(img.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_grad[i, j] = math.sqrt(img_grad_x[i, j] ** 2 + img_grad_y[i, j] ** 2)
    img_grad_x[img_grad_x == 0] = 0.00000001
    angle = img_grad_y / img_grad_x
    plt.subplot(222)
    plt.imshow(img_grad, cmap='gray')
    # 非极大值抑制
    img_nms = np.zeros(img.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_grad[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            elif angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                    flag = False
            if flag:
                img_nms[i, j] = img_grad[i, j]

    plt.subplot(223)
    plt.imshow(img_nms, cmap='gray')

    # 双阈值检测
    lower_boundary = img_grad.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if img_nms[i, j] >= high_boundary:
                img_nms[i, j] = 255
                zhan.append([i, j])
            elif img_nms[i, j] <= lower_boundary:
                img_nms[i, j] = 0
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()
        a = img_nms[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for i in range(temp_1 - 1, temp_1 + 2):
            for j in range(temp_2 - 1, temp_2 + 2):
                if i != j and img_nms[i, j] < high_boundary and img_nms[i, j] > lower_boundary:
                    img_nms[i, j] = 255
                    zhan.append([i, j])

    img_nms[img_nms != 255] = 0

    plt.subplot(224)
    plt.imshow(img_nms, cmap='gray')
    plt.show()
