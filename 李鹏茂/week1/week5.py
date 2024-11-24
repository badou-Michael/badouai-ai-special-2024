import numpy as np
import matplotlib.pyplot as plt
import math




def gaussian_filter(sigma, dim):
    """根据给定的标准差和大小创建高斯滤波器。"""
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))


    return Gaussian_filter / Gaussian_filter.sum()  # 归一化滤波器


def apply_filter(img, filter_kernel):
    """使用卷积将滤波器应用于图像。"""
    dim = filter_kernel.shape[0]  # 获取滤波器的尺寸
    img_pad = np.pad(img, pad_width=((dim // 2, dim // 2), (dim // 2, dim // 2)), mode='constant')  # 填充边缘
    img_filtered = np.zeros_like(img)  # 初始化滤波后的图像

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_filtered[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * filter_kernel)  # 进行卷积操作

    return img_filtered


def compute_gradients(img):
    """使用Sobel算子计算图像的梯度。"""
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel水平算子
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel垂直算子

    img_pad = np.pad(img, pad_width=1, mode='constant')  # 填充边缘
    img_tidu_x = np.zeros_like(img)  # 初始化x方向梯度
    img_tidu_y = np.zeros_like(img)  # 初始化y方向梯度

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向梯度

    img_tidu = np.sqrt(img_tidu_x ** 2 + img_tidu_y ** 2)  # 计算梯度幅值
    img_tidu_x[img_tidu_x == 0] = 1e-10  # 避免除以零
    angle = img_tidu_y/img_tidu_x  # 计算梯度方向

    return img_tidu, angle  # 返回梯度幅值和方向


def non_maximum_suppression(img_tidu, angle):
    img_yizhi = np.zeros(img_tidu.shape)
    """抑制非极大值，保留边缘点。"""
    dx, dy = img.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
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

    return img_yizhi


def double_threshold(img_oxx, img_tidu):
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []

    for i in range(1, img_oxx.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_oxx.shape[1] - 1):
            if img_oxx[i, j] >= high_boundary:  # 取，一定是边的点
                img_oxx[i, j] = 255
                zhan.append([i, j])
            elif img_oxx[i, j] <= lower_boundary:  # 舍
                img_oxx[i, j] = 0

    # 定义相邻像素的相对位置
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]

    while zhan:  # 不需要用 len(zhan) == 0
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_oxx[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]

        # 遍历相邻像素
        for offset in neighbor_offsets:
            neighbor_i = temp_1 + offset[0]
            neighbor_j = temp_2 + offset[1]

            # 确保不超出图像边界
            if 0 <= neighbor_i < img_oxx.shape[0] and 0 <= neighbor_j < img_oxx.shape[1]:
                if lower_boundary < img_oxx[neighbor_i, neighbor_j] < high_boundary:
                    img_oxx[neighbor_i, neighbor_j] = 255  # 这个像素点标记为边缘
                    zhan.append([neighbor_i, neighbor_j])  # 进栈
    for i in range(img_oxx.shape[0]):
        for j in range(img_oxx.shape[1]):
            if img_oxx[i, j] != 0 and img_oxx[i, j] != 255:
                img_oxx[i, j] = 0
    return  img_oxx
    # 绘图]
    # plt.figure(4)
    # plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # plt.axis('off')  # 关闭坐标刻度值
    # plt.show()

if __name__ == '__main__':

    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image",img)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1. 高斯平滑
    sigma = 0.5  # 高斯标准差
    dim = 5  # 高斯滤波器尺寸
    gaussian_filter=gaussian_filter(sigma, dim)
    imgl = apply_filter(img,gaussian_filter)
    img_tidu, angle=compute_gradients(imgl)
    img_jidazhi= np.zeros_like(img_tidu)
    img_jidazhi=non_maximum_suppression(img_tidu, angle)

    img_res = img_jidazhi.copy()  # 使用 .copy() 创建独立副本
    img_res = double_threshold(img_res, img_tidu)
    img_res=double_threshold(img_res,img_tidu)
    # 显示结果
    plt.figure(1)
    plt.imshow(imgl.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
    plt.figure(3)
    plt.imshow(img_jidazhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
    plt.figure(4)
    plt.imshow(img_res.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()

