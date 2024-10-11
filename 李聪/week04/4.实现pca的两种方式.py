# 第一种：实现鸢尾花的 PCA 降维
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据集，x为数据集中的属性数据，y为标签
x, y = load_iris(return_X_y=True)

# 使用PCA将数据降维到2维
pca = PCA(n_components=2)
reduced_x = pca.fit_transform(x)

# 设置存储列表
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

# 根据类别将降维后的数据点存到不同的列表中
for i in range(len(reduced_x)):
    if y[i] == 0:  # 第一类鸢尾花
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:  # 第二类鸢尾花
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:  # 第三类鸢尾花
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 绘制不同类别的散点图
plt.scatter(red_x, red_y, c='r', marker='x', label='Class 0')
plt.scatter(blue_x, blue_y, c='b', marker='D', label='Class 1')
plt.scatter(green_x, green_y, c='g', marker='.', label='Class 2')

# 添加图例
plt.legend()

# 显示图像
plt.show()

# 第二种：实现 lenna 的 PCA 降维
from sklearn.decomposition import PCA
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv.imread("lenna.png", cv.IMREAD_GRAYSCALE)

# 获取图像的维度
n_samples, n_features = image.shape

# 确保 n_components 不超过 min(n_samples, n_features)
n_components = min(50, n_samples, n_features)

# 应用 PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(image)

# 逆变换还原图像
X_pca_inverse = pca.inverse_transform(X_pca)

# 转换为整型显示
X_pca_inverse = np.uint8(X_pca_inverse)

# 显示处理后的图像
plt.imshow(X_pca_inverse, cmap='gray')
plt.show()
