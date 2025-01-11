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
