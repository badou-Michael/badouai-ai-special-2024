import cv2
import random
def  SaltNoise(src,percetage):     #原图，噪声百分比
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
	#在总噪声点中每次取一个随机点
    #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
    #random.randint生成随机整数
	    randX=random.randint(0,src.shape[0]-1)
	    randY=random.randint(0,src.shape[1]-1)
	    #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
	    if random.random()<=0.5:
	    	NoiseImg[randX,randY]=0
	    else:
	    	NoiseImg[randX,randY]=255
    return NoiseImg

img = cv2.imread("lenna.png",0)
img1 = SaltNoise(img,0.75)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Source Image',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)
