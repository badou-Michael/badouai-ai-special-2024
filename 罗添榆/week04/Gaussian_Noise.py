import cv2
import random

#随机生成符合高斯分布的随机数，mean 和 sigma 为两个参数

def GaussianNoise(src,means,sigma,percetage):  #percentage 为 占据原图的比例
    NoiseImg=src        #先将原图赋值到Noise图中
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])   #统计需要添加噪声的总像素点
    for i in range(NoiseNum):
		#在总噪声点中每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上高斯随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        #0~255之间的值保持不变
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

#读取图片
img = cv2.imread('lenna.png',0)
#添加高斯噪声
img1 = GaussianNoise(img,6,8,0.9)
#读图，转换为灰度图
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Gaussian Noise Image",img1)
cv2.imshow("Lenna Source Gray Image",img2)
cv2.imshow("Lenna Source Image",img)
cv2.waitKey(0)
