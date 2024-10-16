#  [模块1:高斯噪声]一:自定义函数添加高斯噪声
import cv2 as cv
# import numpy as np
import random

def GaussianNoise(src, means, sigma, snr):
    NoiseImg = src
    h = NoiseImg.shape[0]
    w = NoiseImg.shape[1]
    NoiseNum = int(h * w * snr)  # 计算 要加噪声的像素个数NoiseNum
    for num in range(NoiseNum):  # 遍历以下代码 NoiseNum 次
        randX = random.randint(0, h - 1)  # 随机获取第 h行数
        randY = random.randint(0, w - 1)  # 随机获取第 w列
        gauss = random.gauss(means, sigma)  # 随机生成一个符合高斯分布的值,函数random.gauss(means, sigma)  means表示均值,sigma表示标准差
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + gauss  # 对图像NoiseImg随机获取的第h行w列的像素值 + 高斯值
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255  # 分别将加噪后的像素值 缩放到[0,255]
        return NoiseImg  # 返回遍历后给NoiseNum个像素加噪后的图像


img1 = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")  # 读取图像
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 获取对应的灰度图方式1
img_gauss1 = GaussianNoise(img, 10, 90, 1)  # 调用函数并将灰度图及相关参数传入
img2 = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png", 0)  # 获取对应的灰度图方式2
img_gauss2 = GaussianNoise(img2, 1, 4, 1)  # 调用函数并将灰度图及相关参数传入
cv.imshow('img', img)
cv.imshow('img_gauss1', img_gauss1)
cv.imshow('img2', img2)
cv.imshow('img_gauss2', img_gauss2)
cv.waitKey(0)

#  [高斯噪声]二_1: 通过在自定义函数中调用numpy.random.normal接口添加高斯噪声--可将输入图像的形状给到目标输出图像noise_img,即可实现直接对彩色图像操作
import cv2 as cv
import random


def GaussianNoise(img, means, sigma):  # means 表示均值,sigma表示标准差
    noise_img = img + np.random.normal(means, sigma, img.shape)
    # 通过np.random.normal(means,sigma,size)函数可生成符合指定均值和标准差的高斯分布的随机数,函数中的3个参数分别表示:means 指均值,sigma 指标准差,size表示形状;
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    # 通过np.clip(目标数组,下限值,上限值)函数,将目标数组中各元素值限制在给定的上下限内,若元素值小于下限值则用下限值替代,高于上限值则用上限值替代
    return noise_img  # 返回加噪后的图像


img = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
gauss = GaussianNoise(img, 0, 25)
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss1 = GaussianNoise(img1, 0, 25)
img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gauss2 = GaussianNoise(img2, 0, 25)
cv.imshow('img', img)
cv.imshow('gauss', gauss)
cv.imshow('img1', img)
cv.imshow('gauss1', gauss)
cv.imshow('img2', img)
cv.imshow('gauss2', gauss)

#   [模块1:高斯噪声]二_2 直接调用np.random.normal()函数,不自定义函数
import cv2 as cv
import numpy as np

img = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
gauss = np.clip(img + np.random.normal(30, 25, img.shape), 0, 255).astype(np.uint8)
cv.imshow('img', img)
cv.imshow('gauss', gauss)

#  [模块1:高斯噪声]三: 使用skimage模块中util中的random_noise函数添加噪声

import numpy as np
from matplotlib import pyplot as plt
from skimage import util

img = plt.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
img = np.array(img).astype(np.float64)
print(img.dtype)
gauss = util.random_noise(img, mode="gaussian", mean=0, var=0.01, clip=True)
gauss1 = util.random_noise(img, mode="gaussian", mean=0.5, var=0.01, clip=True)
gauss2 = util.random_noise(img, mode="gaussian", mean=0, var=4, clip=True)

plt.rcParams['font.sans-serif'] = ['SimHei']    #将plt.title内的中文字体设置为黑体
plt.subplot(221)
plt.title("original")
plt.imshow(img)
plt.subplot(222)
plt.title("gauss:均值mean=0,方差var=0.01")
plt.imshow(gauss)
plt.subplot(223)
plt.title("gauss1:均值mean=0.5,方差var=0.01")
plt.imshow(gauss1)
plt.subplot(224)
plt.title("gauss2:均值mean=0,方差var=9")
plt.imshow(gauss2)
plt.show()

#  测试发现:
# 1.skimage.util.random_noise()函数将图像转换为float64类型(即0-1),而OpenCV处理的图像类型类型是uint8(即0-255),故常用matplotlib模块处理图像(处理后的类型是float32)或from PIL import Image模块
# 2.通过skimage.util.random_noise()函数发现给图像加高斯噪声时,该函数参数的均值mean=0时,加噪后图像亮度跟原图一致,均值>0且越大时,加噪后的图像越亮,直到>=1,加噪后图像整体呈现白色;反之均值<0 且越小时,加噪后图像越暗,直至<= -1,呈现为黑色;
# 3.通过skimage.util.random_noise()函数发现给图像加高斯噪声时,该函数参数的方差var越小,噪声数越小,若var为0,则跟原图对比未加噪声点(加噪后图像只受均值影响亮度有差异),var越大,则噪声点越多,加噪后图像越模糊,在var=0.4时就几乎看不出图像轮廓;

#[模块2:椒盐噪声]一: 根据原理自定义函数实现
import numpy as np
import cv2 as cv
import random
def sp_noise(src, snr):     # 传入图像及噪声比例
    h = src.shape[0]
    w = src.shape[1]
    sp = h * w              # 计算输入图像像素总个数
    np = int(sp * snr)      # 计算要加噪的像素个数
    img_noise = src
    for num in range(np):
        X = random.randint(0, h - 1)
        Y = random.randint(0, w - 1)
        if random.random() >= 0.5:
            img_noise[X, Y] = 255
        else:
            img_noise[X, Y] = 0
    return img_noise

if __name__ == "__main__":
    img =cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
    img1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # img1 = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png", 0)  # 读取lenna的灰度图
    cv.imshow('img', img)
    cv.imshow('img1', img1)
    SpNoise = sp_noise(img1, 0.2)
    cv.imshow('sp_noise', SpNoise)
    cv.waitKey(0)

# [模块2:椒盐噪声] 二: 通过调用skimage.util.noise()函数

import skimage.util as ut
from matplotlib import pyplot as plt

img = plt.imread("E:/GUO_APP/GUO_AI/picture/lenna.png")
img_s_noise = ut.random_noise(img,mode='salt',amount=0.3)    # 添加盐噪声
img_p_noise = ut.random_noise(img,mode='pepper',amount=0.3)  # 添加椒噪声
img_sp_noise = ut.random_noise(img,mode='s&p',amount=0.3,salt_vs_pepper=0.7)   # 添加椒盐噪声

img_s_noise = ut.random_noise(img,mode='poisson')    # 添加泊松噪声
plt.rcParams['font.sans-serif'] = ['SimHei']    #将plt.title内的中文字体设置为黑体
plt.subplot(331)
plt.title('输入图像img')
plt.imshow(img)
plt.subplot(332)
plt.title('盐噪声图:噪声比0.3')
plt.imshow(img_s_noise)
plt.subplot(333)
plt.title('椒噪声图:噪声比0.3')
plt.imshow(img_p_noise)
plt.subplot(334)
plt.title('椒盐噪声图:噪声比0.3,盐椒噪声比0.7')
plt.imshow(img_sp_noise)
plt.subplot(335)
plt.title('泊松噪声图')
plt.imshow(img)
plt.show()

#  [模块3:PCA实现] 一:PCA主成分分析法,根据原理写代码实现
"""
PCA主成分分析方法步骤:
1- 样本矩阵X(M*N)零均值化(即中心化)
2- 计算协方差矩阵  注意,应该对中心化后的样本矩阵求协方差
3- 求解特征值和特征向量
4- 将特征值降序后作为列取前K个特征向量合并--即得到了降维转换矩阵(特征矩阵)
5- 将样本矩阵映射到降维转换矩阵上,即得到了M行K列的降维矩阵
"""
import numpy as np

class PCA(object):                           # 定义类(PCA函数包)
    def __init__(self,X,K):                   # 函数1 :__init__ 初始化函数及定义未知对象的属性(对象未知,约定成俗使用self.属性名)
        self.X=X                             # 样本矩阵X
        self.K=K                              # 矩阵X 的降维矩阵的列数,即要取的特征的个数K
        self.centerX=[]                      # X 中心化后的矩阵
        self.covX=[]                  # X 的协方差矩阵
        self.U = []                       # X 的 前K个特征值对应的特征向量组成的 降维转换矩阵
        self.Z=[]                        # X 的降维矩阵,PCA主成分分析后的输出结果

        self.centerX =self.cent()
        self.covX =self.cov()
        self.U = self._u()
        self.Z = self._Z()
    def cent(self):                         # 函数2: cent()函数,作用是返回 输入矩阵的中心化矩阵
        mean = np.mean(self.X,axis=0)    # np.mean()函数功能,求样本矩阵X的每列的均值(特征均值)
        centerX = self.X -mean       # 中心化样本矩阵X
        print('样本的特征均值mean\n',mean)
        print('样本矩阵X的中心化centerX\n',centerX)
        return centerX

    def cov(self):                # 函数3: cov()作用:返回 输入矩阵X 的协方差矩阵,公式cov=(X.T  * X)/(M-1)  ---矩阵X的转置 乘以 X 再除以(X的总样本数[即行数]-1)
        m = self.centerX[0]      # 样本集的样例总数 (样本矩阵中,一行为一个样例,一列为一个特征)
        covX = np.dot(self.centerX.T,self.centerX) /(m-1)        # 根据公式求得中心化矩阵centerX的协方差矩阵covX
        print(covX)
        return covX

    def _u(self):                               #函数4 ???:_u作用: 计算协方差矩阵的特征值及特征向量,并获取要降维矩阵的K阶转换矩阵
        a,b = np.linalg.eig(self.covX)
        print('特征样本的协方差矩阵covX的特征值\n',a)
        print('特征样本的协方差矩阵covX的特征向量\n',b)
        desc_a=np.argsort(-1* a )
        UT = [b[:,desc_a[i]] for i in range(self.K)]    # ???
        U = np.transpose(UT)
        print('%d阶降维矩阵U\n'%self.K,U)
        return U
    def _Z(self):
        Z = np.dot(self.X,self.U)
        # print('Z的类型\n',type(self.Z))
        # print('Xshape:',np.shape(self.X))
        # print('U shape:',np.shape(self.U))
        # print('Z shape:', np.shape(self.Z))
        # print('样本矩阵X的降维矩阵out:\n',Z)
        print('Z的类型\n', type(self.Z))
        print('Xshape:', self.X.shape)
        print('U shape:', self.U.shape)
        print('Z shape:', np.shape(self.Z))      # 原写法 self.Z.shape,报错"列表中不存在shape",修改为np.shape(self.Z)之后正常运行,
                                                 # 输出Z的类型为列表,为什么呢? Z = X U,得到的也是矩阵才对呀?
        print('样本矩阵X的降维矩阵out:\n', Z)
        return Z

import numpy as np
if __name__ == '__main__':
    X = np.array([[15, 12, 10],
                  [12, 8, 9],
                  [12, 8, 9],
                  [6, 9, 16],
                  [5, 8, 10],
                  [4, 22, 16],
                  [6, 10, 12],
                  [5, 7, 11],
                  [18, 12, 14],
                  [10, 10, 12]])
    K = np.shape(X)[1] - 1
    print(type(X))
    print(X.shape)
    print(X.ndim)
    print('输入的样本矩阵X\n',X)
    pca= PCA(X,K)


#  [模块3:PCA实现] 二:PCA主成分分析法,自定义函数简写版
import numpy as np
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
pca = PCA(n_components=2)
X = np.array([[15, 12, 10],
              [12, 8, 9],
              [12, 8, 9],
              [6, 9, 16],
              [5, 8, 10],
              [4, 22, 16],
              [6, 10, 12],
              [5, 7, 11],
              [18, 12, 14],
              [10, 10, 12]])
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据

#  [模块3:PCA实现] 三:PCA主成分分析法,直接调用接口
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据
