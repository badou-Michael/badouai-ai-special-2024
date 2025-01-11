import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math
import cv2
from skimage.color import rgb2gray

'''
Canny边缘检测算法
1. 对图像进行灰度化
2. 对图像进行高斯滤波：
根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
4 对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点
所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点
5 用双阈值算法检测和连接边缘
'''

if __name__ == '__main__':
    # pic_path = 'lenna.png'
    # img = plt.imread(pic_path)
    # print("img\n", img)
    # if pic_path[-4:] == '.png':
    #     img = img * 255  # 将像素值从 [0, 1] 范围缩放到 [0, 255]
    # img = np.mean(img,axis=-1) # 取均值的方法进行灰度化，axis=-1 表示从最后一个维度计算均值，对于彩色图 (h,w,3) 即通过颜色通道 (R+G+B)/3 计算均值以此来表示灰度值

    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img1",img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    '''
    1. 高斯平滑
    '''
    # sigma = 1.52  高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5
    dim = 5  # 高斯核尺寸 5X5  边缘检测高斯平滑一般选择的尺寸大小是 5X5 比较好
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，是一个数组非列表
    tmp = [i - dim // 2 for i in range(dim)]  # 通过列表表达式生成一个序列  tmp = [-2, -1, 0, 1, 2] 帮助生成以高斯滤波器中心为基准的相对坐标，以便于后面计算高斯核

    # 通过公式计算高斯核
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)

    for i in range(dim):
        for j in range(dim):
            # 以中心为原点(0,0)对应的相对坐标来计算，才符合标准高斯核的对称性。否则按绝对坐标(i,j)直接计算则不符合对称性
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 对高斯核进行归一化处理，使得滤波器的所有元素和为 1 。卷积过程就相当于对原图像进行平滑而不改变平均亮度

    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据

    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补，默认值是0

    # 计算卷积后的值
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)  # 此处滑动步长 stride = 1

    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new的像素值是255的浮点型数据，需要强制转换类型才可以，gray即灰阶
    plt.axis('off')  # 去掉坐标轴

    '''
    2. 求梯度 以下两个滤波是用sobel算子来检测图像中水平、垂直和对角边缘
    '''
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)

    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补

    # 计算水平、垂直方向的卷积结果以及梯度幅值
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = math.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 以防止在求 tanθ 时无值
    angle = img_tidu_y / img_tidu_x

    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''
    3. 非极大值抑制，NMS算法
    '''
    img_yizhi = np.zeros(img_tidu.shape)

    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8领域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8领域矩阵 temp 是当前像素 (i, j) 的 3x3 邻域内的梯度幅值。它用于后续比较
            if angle[i, j] <= -1:  # 使用线性插法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 1] - temp[0, 2]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 0]) / angle[i, j] + temp[2, 1]
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

    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''
    4. 双阈值检测，连接边缘  遍历所有一定是边的点，查看8领域是否存在有可能是边的点，进栈
    
    双阈值检测：
    • 如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；
    • 如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；
    • 如果边缘像素的梯度值小于低阈值，则会被抑制
    '''
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里设置高阈值是低阈值的3倍
    zhan = []

    h, w = img_yizhi.shape

    for i in range(1, h - 1):  # 不考虑外圈
        for j in range(1, w - 1):
            if img_yizhi[i, j] >= high_boundary:  # 强边缘像素保留，一定是边的点
                img_yizhi[i, j] = 255
                # zhan.append([i,j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍弃
                img_yizhi[i, j] = 0
            else:
                zhan.append([i, j])

    # while not len(zhan) == 0:  # 找寻强边缘像素周围8领域是否存在弱边缘像素
    #     temp_1, temp_2 = zhan.pop()  # 出栈
    #     a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    #     if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
    #         img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
    #         zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
    #     if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
    #         img_yizhi[temp_1 - 1, temp_2] = 255
    #         zhan.append([temp_1 - 1, temp_2])
    #     if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
    #         img_yizhi[temp_1 - 1, temp_2 + 1] = 255
    #         zhan.append([temp_1 - 1, temp_2 + 1])
    #     if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
    #         img_yizhi[temp_1, temp_2 - 1] = 255
    #         zhan.append([temp_1, temp_2 - 1])
    #     if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
    #         img_yizhi[temp_1, temp_2 + 1] = 255
    #         zhan.append([temp_1, temp_2 + 1])
    #     if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
    #         img_yizhi[temp_1 + 1, temp_2 - 1] = 255
    #         zhan.append([temp_1 + 1, temp_2 - 1])
    #     if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
    #         img_yizhi[temp_1 + 1, temp_2] = 255
    #         zhan.append([temp_1 + 1, temp_2])
    #     if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
    #         img_yizhi[temp_1 + 1, temp_2 + 1] = 255
    #         zhan.append([temp_1 + 1, temp_2 + 1])

    while not len(zhan) == 0:  # 找寻强边缘像素周围8领域是否存在弱边缘像素
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if a[0, 0] == 255 or a[0, 1] == 255 or a[0, 2] == 255 or a[1, 0] == 255 or a[1, 2] == 255 or a[2, 0] == 255 or \
                a[2, 1] == 255 or a[2, 2] == 255:
            img_yizhi[temp_1, temp_2] = 255

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
