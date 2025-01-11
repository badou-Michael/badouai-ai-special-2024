import numpy as np
import cv2
import random


# 实现高斯噪声
def GaussianNoise(src, means, sigma, precetage):
    noise_img = src.copy()

    # 获取需要增加高斯噪声的像素点数量
    noise_num = int(precetage * src.shape[0] * src.shape[1])

    # 循环给每一个像素点增加高斯噪声的像素值
    for i in range(noise_num):
        # 获取增加高斯噪声的像素点坐标
        # randX表示行 H，randY表示列 W
        # 其中高斯噪声图片边缘不做处理，需要 -1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # 原有像素灰度值基础上加上随机数
        noise_img[randX,randY] = noise_img[randX,randY] + random.gauss(means,sigma)

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if noise_img[randX,randY] < 0:
            noise_img[randX, randY] = 0
        elif noise_img[randX,randY] > 255:
            noise_img[randX, randY] = 255

    return noise_img

# 效果展示
'''
cv2.imread说明：
第二个参数值：
不填或1 表示以彩色图像格式加载图像（默认）
0 表示以灰度图像格式加载图像
-1 加载图像，并保留其原始Alpha通道（如果存在）
'''
img = cv2.imread('lenna.png',0)

noise_img = GaussianNoise(img,2,4,0.8) # 获取高斯噪声后图片

combined_img = np.hstack([img,noise_img])
cv2.imshow('combined_img',combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#################################################################################################
# 彩色图像实现高斯噪声
def GaussianNoise(src, means, sigma, precetage):
    noise_img = src.copy()

    # 获取需要增加高斯噪声的像素点数量
    noise_num = int(precetage * src.shape[0] * src.shape[1])

    # 通道数
    channels = src.shape[2]

    for c in range(channels):

        # 循环给每一个像素点增加高斯噪声的像素值
        for i in range(noise_num):
            # 获取增加高斯噪声的像素点坐标
            # randX表示行 H，randY表示列 W
            # 其中高斯噪声图片边缘不做处理，需要 -1
            randX = random.randint(0, src.shape[0] - 1)
            randY = random.randint(0, src.shape[1] - 1)

            # 原有像素灰度值基础上加上随机数
            noise_img[randX,randY,c] = noise_img[randX,randY,c] + random.gauss(means,sigma)

            # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
            if noise_img[randX,randY,c] < 0:
                noise_img[randX, randY,c] = 0
            elif noise_img[randX,randY,c] > 255:
                noise_img[randX, randY,c] = 255

    return noise_img

# 效果展示
'''
cv2.imread说明：
第二个参数值：
不填或1 表示以彩色图像格式加载图像（默认）
0 表示以灰度图像格式加载图像
-1 加载图像，并保留其原始Alpha通道（如果存在）
'''
img = cv2.imread('lenna.png')

noise_img = GaussianNoise(img,2,4,0.8) # 获取高斯噪声后图片

combined_img = np.hstack([img,noise_img])
cv2.imshow('combined_img',combined_img)

cv2.waitKey(0)
cv2.destroyAllWindows()





