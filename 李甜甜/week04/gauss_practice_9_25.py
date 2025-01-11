'''高斯噪声， 噪声分布服从高斯分布， 添加高斯噪声
一:   确定高斯函数（参数）
二：确定噪声添加数量num（比例）
三：重复num次，随机挑出某列某行像素，添加在原本像数值基础上添加高斯随机数
'''
import random
import cv2
import numpy
#高斯噪音函数需要获取四个参数， 原图像，高斯参数两个，噪音比例
def GaussionNoise(src_img,means,sigma,percentage):
    # 不能在原图像上进行更改，需要再建立一个噪音图
    GaussImg =src_img
    #确定添加比例，最后数值需要变成整型
    GaussNum =int(percentage*src_img.shape[0]*src_img.shape[1])
    #开始添加噪音
    for i in range(GaussNum):
        #随机行列添加
        X= random.randint(0,src_img.shape[0]-1)
        Y= random.randint(0,src_img.shape[1]-1)
        GaussImg[X, Y] = GaussImg[X,Y] + random.gauss(means,sigma)
        if GaussImg[X, Y] >255:
            GaussImg[X, Y] = 255
        elif GaussImg[X, Y] <0:
            GaussImg[X, Y] = 0
    return GaussImg

img0 = cv2.imread("lenna.png",0)
img1 = GaussionNoise(img0, 2, 4, 0.8)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img_gray", img2)
cv2.imshow("Gaussimg", img1)

print("src", img2)
print("gauss", img1)
cv2.waitKey(0)
