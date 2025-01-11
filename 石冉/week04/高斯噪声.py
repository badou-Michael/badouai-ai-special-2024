#实现高斯噪声
import random
import cv2


#定义高斯噪声函数
def GaussNoise(src,mean,sigma,percentage):
    NoiseImg=src   #将原始图片复制，在复制图片上修改
    NoiseNum=int(percentage*src.shape[0]*src.shape[1]) #计算添加噪声的像素点总数
    for i in range(NoiseNum):
        # 随机生成需要添加噪声的坐标
        randx=random.randint(0,src.shape[0]-1)
        randy=random.randint(0,src.shape[1]-1)
        #添加高斯噪声值
        NoiseImg[randx,randy]=NoiseImg[randx,randy]+random.gauss(mean,sigma)
        #把0-255区间外的数修改为0或255
        if NoiseImg[randx,randy]<0:
            NoiseImg[randx,randy]=0
        elif NoiseImg[randx,randy]>255:
            NoiseImg[randx,randy]=255
    return NoiseImg

img=cv2.imread('lena.png',0)
img1=GaussNoise(img,8,10,1)
img=cv2.imread('lena.png',0)
cv2.imshow('source',img)
cv2.imshow('gaussnoise',img1)
cv2.waitKey(0)
