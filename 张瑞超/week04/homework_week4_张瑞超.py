#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape
import random
from skimage import util

# Implementation of gauss noise and pepper salt noise
def add_noise(src, noise_type='gaussian', mean=0, sigma=25, percentage=0.05):
    """
    为图像添加噪声。

    参数:
    src (np.ndarray): 输入的原图像。
    noise_type (str): 噪声类型，'gaussian' 为高斯噪声，'salt_pepper' 为椒盐噪声。
    mean (float): 高斯噪声的均值。
    sigma (float): 高斯噪声的标准差。
    percentage (float): 添加噪声的比例，默认是 0.05。

    返回:
    np.ndarray: 添加噪声后的图像。
    """
    # 使用 np.copy() 确保不会改变原图像
    noise_img = np.copy(src)
    noise_num = int(percentage * src.shape[0] * src.shape[1])

    for _ in range(noise_num):
        # 随机生成像素坐标
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)

        # 根据噪声类型处理不同的噪声
        if noise_type == 'gaussian':
            # 添加高斯噪声
            noise_value = noise_img[rand_x, rand_y] + random.gauss(mean, sigma)
        elif noise_type == 'salt_pepper':
            # 添加椒盐噪声，随机选择 0（黑点）或 255（白点）
            noise_value = 0 if random.random() <= 0.5 else 255
        else:
            raise ValueError("Invalid noise_type. Choose 'gaussian' or 'salt_pepper'.")

        # 限制噪声后的像素值在 [0, 255] 范围内
        noise_img[rand_x, rand_y] = np.clip(noise_value, 0, 255)

    return noise_img

# 自己手写
img = cv2.imread('lenna.png',0)
img1 = add_noise(img, 'gaussian',2,4,0.8)
img3 = add_noise(img, 'salt_pepper',percentage=0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('grey source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.imshow('lenna_PepperSaltNoise',img3)


# 调用接口
img4=util.random_noise(img,mode='poisson')
cv2.imshow('original source',img)
cv2.imshow('lenna_PoissonNoise',img4)



# Implementation of PCA
class PCA:
    def __init__(self, n_components):
        # n_components 指的是降维后的维度数
        self.n_components = n_components
        # 前 n_components 个特征向量，初始化为None
        self.components_ = None

    def fit_transform(self, x):
        # 将数据进行中心化（零均值）
        x_centered = x - np.mean(x, axis=0)

        # 计算协方差矩阵
        covariance_matrix = np.cov(x_centered, rowvar=False)

        # 计算协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eigh(covariance_matrix)

        # 对特征值进行降序排序，获取排序后的索引
        sorted_idx = np.argsort(eig_vals)[::-1]
        sorted_eig_vectors = eig_vectors[:, sorted_idx]

        # 选择前 n_components 个特征向量
        self.components_ = sorted_eig_vectors[:, :self.n_components]

        # 将数据投影到新的特征空间中，实现降维
        x_reduced = np.dot(x_centered, self.components_)

        return x_reduced


pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX = pca.fit_transform(X)
print(f'降维后的矩阵：\n{newX}')


cv2.waitKey(0)