# 1.实现高斯噪声、椒盐噪声手写实现
# 2.噪声接口调用
# 3.实现pca
import random
from statistics import stdev

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.core import shape
from skimage.util import random_noise

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        #去中心化
        self.mean = np.mean(X, axis=0)
        standardized_X = X - self.mean

        #协方差矩阵
        covariance_matrix = np.cov(standardized_X, rowvar=False)

        #特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        #按特征值降序排序特征向量
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        sorted_eigenvalues = eigenvalues[sorted_index]

        #选择指定数量的成分的特征向量
        self.components = sorted_eigenvectors[:, 0:self.n_components]

        #转换数据
        self.transformed_data = np.dot(standardized_X, self.components)

        return self

    def fit_transform(self, X):
        return self.fit(X).transformed_data

    def transform(self, X):
        standardized_X = X - self.mean
        return np.dot(standardized_X, self.components)

# 图像矩阵 均值 标准差
def add_gaussian_noise(image, mean, std_dev):
    guass_noise = np.random.normal(mean, std_dev, image.shape)
    new_image = image + guass_noise
    new_image = np.clip(new_image + 0.5, 0, 255).astype('uint8')
    return new_image

# 图像矩阵、信噪比
def add_salt_paper_noise(image, SNR):
    h, w, c = image.shape
    total_pixels = h * w * c
    num_noise_pixels = int(total_pixels * SNR)
    noise_indices = np.random.choice(total_pixels, num_noise_pixels, replace=False)
    flat_image = image.reshape(-1)
    for index in noise_indices:
        flat_image[index] = 0 if random.random() > 0.5 else 255
    return flat_image.reshape(h, w, c)

image = cv2.cvtColor(cv2.imread("lenna.png"), cv2.COLOR_BGR2RGB)

# 设置字体为 SimHei
plt.rcParams['font.family'] = 'SimHei'
# 创建一个 2x2 的子图
fig, axs = plt.subplots(3, 2, figsize=(8, 8))
# 将所有子图的坐标轴关闭
for ax in axs.flat:
    ax.axis('off')

# 在每个子图上显示图片
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('原图')

#问题：自己写的和skimage库里的生成的噪声图片差别很大
axs[1, 1].imshow(add_gaussian_noise(image, 0, 0.1))
axs[1, 1].set_title('高斯噪声')

axs[2, 1].imshow(add_salt_paper_noise(image, 0.1))
axs[2, 1].set_title('椒盐噪声')

axs[1, 0].imshow((random_noise(image, mode='gaussian', var=0.01, mean=0) * 255).astype(np.uint8))
axs[1, 0].set_title('高斯噪声')

axs[2, 0].imshow((random_noise(image, mode='s&p', amount=0.1) * 255).astype(np.uint8))
axs[2, 0].set_title('椒盐噪声')


print('np.uint8: ', np.uint8)
plt.show()
cv2.waitKey(0)

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
pca=PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
reduced_x=PCA.fit_transform(pca, x) #对原始数据进行降维，保存在reduced_x中



red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()