import numpy as np
import matplotlib.pyplot as plt
import math

import cv2

def read_and_grayscale(pic_path):
    """读取图像并转换为灰度图像"""
    try:
        img = plt.imread(pic_path)
        if pic_path[-4:] == '.jpg':
            img = img * 255
        img = img.mean(axis=-1)
        return img
    except FileNotFoundError:
        print(f"文件 {pic_path} 未找到")
        return None
    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return None


def gaussian_smoothing(img, sigma=0.5, dim=5):
    """高斯平滑"""
    Gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    return img_new


def compute_gradients(img):
    """计算梯度"""
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros(img.shape)
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    dx, dy = img.shape  # 获取图像的维度
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    return img_tidu, angle


def non_max_suppression(img_tidu, angle):
    """非极大值抑制"""
    img_yizhi = np.zeros(img_tidu.shape)
    dx, dy = img_tidu.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] <= -1:
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


def double_threshold(img_yizhi, lower_boundary, high_boundary):
    """双阈值检测和边缘连接"""
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0

    while zhan:
        temp_1, temp_2 = zhan.pop()
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for di in range(3):
            for dj in range(3):
                if (a[di, dj] < high_boundary) and (a[di, dj] > lower_boundary):
                    img_yizhi[temp_1 + di - 1, temp_2 + dj - 1] = 255
                    zhan.append([temp_1 + di - 1, temp_2 + dj - 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    return img_yizhi


if __name__ == '__main__':
    pic_path = 'lenna.jpg'
    img = read_and_grayscale(pic_path)
    print(img.shape)

    # 实现canny的详细步骤
    if img is not None:
        img_smoothed = gaussian_smoothing(img)
        img_tidu, angle = compute_gradients(img_smoothed)
        img_yizhi = non_max_suppression(img_tidu, angle)
        lower_boundary = img_tidu.mean() * 0.5
        high_boundary = img_tidu.mean() * 1.5
        img_result = double_threshold(img_yizhi, lower_boundary, high_boundary)
        cv2.imshow("canny1", img_result)

    # 直接调用cv2。Canny实现
    img2 = cv2.imread("lenna.jpg", 1)
    image2 = cv2.Canny(img2, 160, 180)
    cv2.imshow("canny2", image2)
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()