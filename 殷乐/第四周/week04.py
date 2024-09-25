import random
import cv2
from skimage import util
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import eig


def add_gaussian_noise(org_image, mean, var, ratio):
    noise_img = org_image
    # 获取噪声像素点数量
    noise_num = int(ratio*noise_img.shape[0]*noise_img.shape[1])
    # 遍历
    for i in range(noise_num):
        # 随机X和Y坐标
        x = random.randint(0, noise_img.shape[0]-1)
        y = random.randint(0, noise_img.shape[1]-1)
        # 添加高斯噪声
        noise_img[x, y] = noise_img[x, y] + random.gauss(mean, var)
        # 判断像素值是否在0-255
        if noise_img[x, y] < 0:
            noise_img[x, y] = 0
        elif noise_img[x, y] > 255:
            noise_img[x, y] = 255
    return noise_img


def add_sp_noise(org_image, ratio):
    noise_img = org_image
    # 获取噪声像素点数量
    noise_num = int(ratio * noise_img.shape[0] * noise_img.shape[1])
    # 遍历
    for i in range(noise_num):
        # 随机X和Y坐标
        x = random.randint(0, noise_img.shape[0] - 1)
        y = random.randint(0, noise_img.shape[1] - 1)
        # 生成0-1之间随机浮点数，判断添加椒噪声还是盐噪声
        if random.random() < 0.5:
            noise_img[x, y] = 0
        else:
            noise_img[x, y] = 255
    return noise_img


def use_api(org_image):
    cv2.imshow('lenna', org_image)
    gaussian_img_api = util.random_noise(image, mode='gaussian', mean=0, var=0.01)
    cv2.imshow('lenna_GaussianNoise_api', gaussian_img_api)
    sp_image_api = util.random_noise(image, mode='s&p')
    cv2.imshow('lenna_spNoise_api', sp_image_api)


def gaussian():
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('lenna_source', img_gray)
    gaussian_img = add_gaussian_noise(img_gray, 2, 10, 1)
    cv2.imshow('lenna_GaussianNoise', gaussian_img)


def sap():
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('lenna_source', img_gray)
    sp_image = add_sp_noise(img_gray, 0.8)
    cv2.imshow('lenna_spNoise', sp_image)


# k：降维后的阶数
def pca(k):
    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    # print(iris.data.shape)
    X = iris.data
    # 去中心化
    # print(X.mean(axis=0))
    X = X - X.mean(axis=0)
    # 求样本矩阵的协方差矩阵
    C = np.cov(X, rowvar=False)
    # 计算协方差矩阵的特征值
    eig_values, eig_vectors = eig(C)
    # 特征值按大小排序
    eig_values_sorted = np.sort(eig_values)[::-1]
    # print(eig_values_sorted)
    # 获得降序排列特征值的序号
    idx = np.argsort(eig_values_sorted)
    # 特征向量按对应特征值排序
    eig_vectors_sorted = eig_vectors[:, idx[:k]]
    # print(eig_vectors_sorted)
    # 得到降维后的矩阵
    new_data = np.dot(X, eig_vectors_sorted)
    print(new_data.shape)

# 读图
image = cv2.imread("lenna.png")
# 1.添加高斯噪声
# gaussian()

# 2.添加椒盐噪声
# sap()

# 3.调用接口
# use_api(image)
# cv2.waitKey(0)

# 4.实现pca
pca(2)
