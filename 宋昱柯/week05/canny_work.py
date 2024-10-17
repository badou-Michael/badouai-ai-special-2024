import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# 设置参数
sigma = 1
dim = 5
low_ratio = 0.5
high_ratio = 1.5


def read_img(pic_path):
    """读取RGB图像"""
    img = cv.imread(pic_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def to_gray(img):
    """图像灰度化"""
    return np.dot(img[..., :3],[0.299, 0.587, 0.114])


def gauss_filter(img, sigma, dim):
    """高斯平滑"""
    kernel = np.zeros((dim, dim))
    n1 = 1 / (2 * np.pi * sigma**2)
    n2 = -1 / (2 * sigma**2)

    for i in range(dim):
        for j in range(dim):
            kernel[i, j] = n1 * np.exp(
                n2 * ((i - dim // 2) ** 2 + (j - dim // 2) ** 2))

    img_new = np.zeros(img.shape)
    img_pad = np.pad(img, dim // 2, "constant")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_new[i, j] = np.sum(img_pad[i : i + dim, j : j + dim] * kernel)

    return img_new


def cal_grad_and_dirction(img):
    """计算图像像素梯度幅值和梯度方向"""
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = np.zeros(img.shape)
    grad_y = np.zeros(img.shape)
    img_grad = np.zeros(img.shape)
    img_pad = np.pad(img, 1, "constant")

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grad_x[i, j] = np.sum(img_pad[i : i + 3, j : j + 3] * sobel_kernel_x)
            grad_y[i, j] = np.sum(img_pad[i : i + 3, j : j + 3] * sobel_kernel_y)
            img_grad[i, j] = np.sqrt(grad_x[i, j] ** 2 + grad_y[i, j] ** 2)
    grad_x[grad_x == 0] = 1e-8
    dirction = grad_y / grad_x

    return img_grad, dirction


def suppress_linear(img,img_grad,direction):
    """非极大值抑制"""
    img_suppress = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            flag = False
            neibor = img_grad[i - 1 : i + 2, j - 1 : j + 2]
            if direction[i, j] <= -1:
                c1 = (neibor[0, 1] - neibor[0, 0]) / direction[i, j] + neibor[0, 1]
                c2 = (neibor[2, 1] - neibor[2, 2]) / direction[i, j] + neibor[2, 1]
                if img_grad[i, j] > c1 and img_grad[i, j] > c2:
                    flag = True
            elif direction[i, j] >= 1:
                c1 = (neibor[0, 2] - neibor[0, 1]) / direction[i, j] + neibor[0, 1]
                c2 = (neibor[2, 0] - neibor[2, 1]) / direction[i, j] + neibor[2, 1]
                if img_grad[i, j] > c1 and img_grad[i, j] > c2:
                    flag = True
            elif direction[i, j] >= 0:
                c1 = (neibor[0, 2] - neibor[1, 2]) * direction[i, j] + neibor[1, 2]
                c2 = (neibor[2, 0] - neibor[2, 1]) * direction[i, j] + neibor[1, 0]
                if img_grad[i, j] > c1 and img_grad[i, j] > c2:
                    flag = True
            elif direction[i, j] < 0:
                c1 = (neibor[1, 0] - neibor[0, 0]) * direction[i, j] + neibor[1, 0]
                c2 = (neibor[1, 2] - neibor[2, 2]) * direction[i, j] + neibor[1, 2]
                if img_grad[i, j] > c1 and img_grad[i, j] > c2:
                    flag = True

            if flag:
                img_suppress[i, j] = img_grad[i, j]

    return img_suppress


def double_threshold(img_suppress, img_grad, low_ratio=0.5, high_ratio=1.5):
    """双阈值检测"""
    thesh_low = img_grad.mean() * low_ratio
    thesh_high = img_grad.mean() * high_ratio

    new_img = np.zeros_like(img_suppress)

    strong_edge = img_suppress >= thesh_high
    weak_edge = (img_suppress >= thesh_low) & (img_suppress < thesh_high)
    new_img[strong_edge] = 255

    # 8邻域偏移
    offset = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # 筛选出强边缘的坐标
    loc = list(zip(*np.where(strong_edge)))
    while loc:
        # y, x分别代表像素坐标y轴与x轴
        y, x = loc.pop()
        for dx, dy in offset:
            if 0 <= y + dy < new_img.shape[0] and 0 <= x + dx < new_img.shape[1]:
                if weak_edge[y + dy, x + dx] and new_img[y + dy, x + dx] == 0:
                    new_img[y + dy, x + dx] = 255
                    loc.append((y + dy, x + dx))

    return new_img


def canny_algorithm(pic_path):
    """canny主函数"""
    img = read_img(pic_path)
    new_img = img.copy()
    
    new_img = to_gray(new_img)
    new_img = gauss_filter(new_img,sigma,dim)
    img_grad,direction = cal_grad_and_dirction(new_img)
    new_img = suppress_linear(new_img,img_grad,direction)
    new_img = double_threshold(new_img,img_grad,low_ratio,high_ratio)
    
    return new_img


if __name__ == '__main__':
    pic_path = "practice\cv\week05\lenna.png"
    new_img = canny_algorithm(pic_path)
    cv.imshow("new img",new_img.astype(np.uint8))
    cv.waitKey(0)