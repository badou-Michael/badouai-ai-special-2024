import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

if __name__ == '__main__':
    # 图像的基本处理
    img = plt.imread("lenna.png")
    pic_path = 'lenna.png'
    if pic_path[-4:] == '.png':     # plt读取png格式图像时像素值为0-1的浮点数
        img = img * 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯滤波
    sigma = 0.5
    dim = int(np.round(6 *sigma + 1))
    if dim % 2 == 0:
        dim = dim + 1
    Gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)  # n1：计算高斯核的归一化因子
    n2 = -1 / (2 * sigma ** 2)  # n2：计算高斯函数的负二次项系数
    # 生成dim*dim个高斯权值
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 归一化处理，否则图像会偏亮或偏暗
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')     # 卷积前边缘填报以保证滤波后图像大小不变
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter)     # 加权求和
    
    # Sobel算子计算梯度幅值和方向
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j] ** 2 + img_tidu_y[i,j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.0000000000000001
    angle = img_tidu_y / img_tidu_x
    
    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True # True-是极大值保留像素点、False-不是极大值要抑制
            temp = img_tidu[i-1:i+2, j-1:j+2]   # 8领域
            # 线性插值计算梯度正反方向上的两个虚拟点的梯度幅值
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # False标记为抑制该像素点img_tidu[i, j]
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]

    plt.subplot(221)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.subplot(222)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.subplot(223)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')

    # 双阈值检测并连接边缘
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 强边缘点的判定
                img_yizhi[i, j] = 255
                zhan.append([i, j])  # 将强边缘的坐标加入栈zhan中，使用栈zhan来跟踪已知的强边缘点
            elif img_yizhi[i, j] <= lower_boundary:  # 若边缘点
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        # 强边缘点的8领域检查：如果邻域中的某个像素的灰度值在高阈值和低阈值之间（弱边缘点），则将该像素标记为边缘（值设为255），并将其坐标加入栈中以便进一步连接
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            # 双阈值检测就是强弱边缘连接 ，新的弱边缘变成强边缘以后也得入栈判断周围是否有弱边缘从而连接
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.subplot(224)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.show()
