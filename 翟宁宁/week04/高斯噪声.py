'''
实现高斯噪声
高斯分布的means,sigma为两个参数
如何来人为的增加噪声，让数据（图像像素）服从高斯分布，把图像上
每个像素点作用一个高斯随机数
'''

import random
import  cv2 as cv


def gauss_noise_functon(img, sigma, means, percent):
    gauss_noise_img = img
    gauss_noise_num= int(percent*img.shape[0]*img.shape[1])
    for i in range(gauss_noise_num):
        # 变量每个像素，添加高斯随机数
        #随机生成h大小随机数
        ranX = random.randint(0,img.shape[0]-1)
        ranY = random.randint(0,img.shape[1]-1)
        rand_g = random.gauss(means, sigma)
        gauss_noise_img[ranX,ranY] = img[ranX,ranY]+rand_g
        print('rand_G = %d'%rand_g)
        #边界
        if gauss_noise_img[ranX,ranY] < 0:
            gauss_noise_img[ranX,ranY] = 0
        elif gauss_noise_img[ranX,ranY] > 255:
            gauss_noise_img[ranX,ranY] = 255

    return gauss_noise_img

img = cv.imread("./images/lenna.png",cv.IMREAD_GRAYSCALE)
noise_img = gauss_noise_functon(img,64,128,0.5)
img1 = cv.imread("./images/lenna.png",cv.IMREAD_GRAYSCALE)
cv.imshow("noise image",noise_img)
cv.imshow("main image",img1)
cv.waitKey(0)
