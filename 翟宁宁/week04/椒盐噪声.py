'''
实现椒盐噪声
椒盐噪声表现为黑白 ，和之前的二值化有点相似
'''

import random
import  cv2 as cv


def pepper_salt_functon(img, percent):
    pepper_salt_img = img
    pepper_salt_num= int(percent*img.shape[0]*img.shape[1])
    for i in range(pepper_salt_num):
        #随机生成h大小随机数
        ranX = random.randint(0,img.shape[0]-1)
        ranY = random.randint(0,img.shape[1]-1)
        #rand_g = random.gauss(means, sigma)
        #pepper_salt_img[ranX,ranY] = img[ranX,ranY]+rand_g
        #print('rand_G = %d'%rand_g)
        randn = random.random()  # 模拟一个数（0-1)，来表示当前坐标像素值 （0或255）
        print(randn)
        if randn <= 0.5:
            pepper_salt_img[ranX,ranY] = 0
        else:
            pepper_salt_img[ranX,ranY] = 255

    return pepper_salt_img
img = cv.imread("./images/lenna.png",cv.IMREAD_GRAYSCALE)
pepper_salt_img = pepper_salt_functon(img,0.5)
img1 = cv.imread("./images/lenna.png",cv.IMREAD_GRAYSCALE)
cv.imshow("noise image",pepper_salt_img)
cv.imshow("main image",img1)
cv.waitKey(0)
