# 使用skimage库中的函数为图像添加高斯噪声

'''
函数式：skimage.util.random_noise(image, mode=‘gaussian’, seed=None, clip=True, **kwargs)
参数介绍：
image：输入图像，类型为ndarray；
mode：噪声类别，有以下几种：
    'gaussian'：高斯噪声；
    'localvar'：高斯分布的加性噪声，在图像每个点都有特点的局部方差；
    'poisson'：泊松噪声；
    'salt'：盐噪声，随机将图像像素值变为1；
    'pepper'：椒噪声，随机将图像像素值变为0或 - 1；
    's&p'：椒盐噪声；
    'speckle'：均匀噪声，（均值mean方差variance），out = image + n * image，n是具有指定均值和方差的均匀噪声；
seed：可选，int型，如果选择的话，则会在生成噪声前设置随机种子；
clip：可选，bool型，若为True(default)则在加入‘speckle’, ‘poisson’, 或 ‘gaussian’这三种噪声后，
    进行剪切以保证图像数据点都在[0, 1]或[-1.1]之间。若为False，则数据可能超出这个范围；
mean：可选，float型，用于’gaussian’和‘speckle’的均值设置，默认为0；
var：可选，float型，用于’gaussian’和‘speckle’的方差设置，默认为0.01；
local_vars：可选，ndarray型，用于‘localvar’的图像每个像素点处的局部方差设置；
amount：可选，float型，用于‘salt’, ‘pepper’和‘s & p’的噪声比例，默认为0.05；
salt_vs_pepper：可选，float型，用于's&p'中盐噪声与椒噪声的比例，范围为[0, 1]，默认为0.5；
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
'''

from skimage import util
import cv2
import numpy as np

img = cv2.imread('lenna.png')    # 读取lenna彩色图片
noise_gs_img = util.random_noise(img, mode='gaussian', mean=0, var=0.1)   # 调用高斯噪声
noise_jy_img = util.random_noise(img, mode='s&p', amount=0.1)   # 调用椒盐噪声

# 画图，原图，高斯噪声图，椒盐噪声图
cv2.imshow("source", img)
cv2.imshow("lenna_Noise_gs", noise_gs_img)
cv2.imshow("lenna_Noise_jy", noise_jy_img)
cv2.waitKey(0)