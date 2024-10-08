
import numpy as np
from sklearn.decomposition import PCA

# 导入数据，维度为人(年龄，资金，性别)
X = np.array([[20,2000,1], [18,500,1], [28,300000,1], [29,200000,0], [27,100000,0], [19,3000,0]])
'''矩阵X的中心化'''
print('样本矩阵X:\n', X)

mean = np.array([np.mean(attr) for attr in X.T])  # 样本集的特征均值
print('样本集的特征均值:\n',mean)
centrX = X - mean #去中心化
print('样本矩阵X的中心化centrX:\n', centrX)
pca = PCA(n_components=3)   #降到3维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据



# 导入数据，维度为新能源汽车(行驶里程，额定容量，soh，出厂年份)
X = np.array([[9874,298,90,2022], [12588,204,89,2022], [84778,152.1,89,2021], [55073,257,89,2021], [5385,52,90,2023], [809,303,99,2022]])
pca = PCA(n_components=3)   #降到3维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=3)
X = np.array([[9874,298,90,2022], [12588,204,89,2022], [84778,152.1,89,2021], [55073,257,89,2021], [5385,52,90,2023], [809,303,99,2022]])
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from skimage import io, img_as_float
# from skimage.util import random_noise
# from skimage.transform import resize
#
# # 读取图像并转换为浮点型
# image = img_as_float(io.imread('sea.jpg', as_gray=True))
#
# # 添加噪声
# noisy_image = random_noise(image, var=0.05)  # 添加高斯噪声
#
#
# # 缩小图像尺寸以提高处理速度
# noisy_image_resized = resize(noisy_image, (noisy_image.shape[0] // 2, noisy_image.shape[1] // 2), anti_aliasing=True)
#
# # 显示原始图像和噪声图像
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Noisy Image')
# plt.imshow(noisy_image, cmap='gray')
# plt.axis('off')
# plt.show()
#
# # 将图像重塑为二维数据
# h, w = noisy_image.shape
# noisy_image_reshaped = noisy_image.reshape(h * w, 1)
#
# # 获取样本数量和特征数量
# n_samples, n_features = noisy_image_reshaped.shape
#
# # 使用 PCA 降噪
# # 使用 PCA 降噪，确保 n_components 合理
# n_components = min(n_samples, n_features) - 1  # 设置为样本数和特征数中的较小值减去1
# pca = PCA(n_components)  # 选择主成分数量
# pca.fit(noisy_image_reshaped)
# denoised_image_reshaped = pca.inverse_transform(pca.transform(noisy_image_reshaped))
#
# # 重塑为图像形状
# denoised_image = denoised_image_reshaped.reshape(h, w)
#
# # 显示降噪后的图像
# plt.figure(figsize=(6, 6))
# plt.title('Denoised Image using PCA')
# plt.imshow(denoised_image, cmap='gray')
# plt.axis('off')
# plt.show()
# input("Press Enter to continue...")  # 等待用户输入以保持窗口打开
