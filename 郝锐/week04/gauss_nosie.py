#图像添加高斯噪声
import random

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('lenna.jpg')
#添加高斯噪声
def Gauss_noise(src,mean,sigma,percentage):
    """

    :param src:源图像
    :param mean:均值：高斯分布的中心位置，表示分布的集中趋势
    :param sigma:方差：衡量了数据相对于均值的分散程度，标准差越大，数据点越分散；标准差越小，数据点越集中在均值附近。
    :param percentage: 控制噪声强度，小-图像影响小，大-图像失真严重
    :return: :param img: 添加高斯噪声后的图像
    """
    noise_img = src
    noise_num = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noise_num): #随机取一点
        #随机取src上一个像素，除去边缘像素（-1）
        randY = random.randint(0,src.shape[0]-1)
        randX = random.randint(0,src.shape[1]-1)
        #在原有灰度图上像素对应的灰度值添加随机数
        #random.gauss(mean,sigma)复制生成随机高斯数
        noise_img[randY,randX] = noise_img[randY,randX] + random.gauss(mean,sigma)
        # 若灰度值为0到255，所以计算后有能超出这个范围，则强制归0或者255
        if noise_img[randY,randX] < 0:
            noise_img[randY,randX] = 0
        elif noise_img[randY,randX] > 255:
            noise_img[randY,randX] = 255
    return noise_img
img = cv2.imread("lenna.png",0) #0，表示读取灰度图
img_gauss = Gauss_noise(img,0,10,0.9)
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('original image')
plt.subplot(122),plt.imshow(img_gauss,cmap='gray'),plt.title('gauss noise image')
plt.show()
