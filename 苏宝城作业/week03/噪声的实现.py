#导入库
import numpy as np
import cv2 as cv
import random

#高斯噪声函数定义
def GaoSiNoise(src, means, sigma, percetage):  #参数分别为传入的图像、加入的高斯噪声值的分布期望、sima大小以及百分比）
  img = src
  noise_num = int(percetage * src.shape[0] * src.shape[1])
  for i in range(noise_num):
    randX = random.randint(0, src.shape[0] - 1)
    randY = random.randint(0, src.shape[1] - 1)
    #上面为随机赋予的坐标值
    img[randX, randY] += random.gauss(means, sigma)

    #阈值处理：
    if img[randX, randY] < 0:
      img[randX, randY] = 0
    elif img[randX, randY] > 255:
      img[randX, randY] = 255
      return img

#椒盐噪声函数定义
def JiaoYanNoise(src,  percetage):
  img = src
  noise_num = int(percetage * src.shape[0] * src.shape[1])
  for i in range(NoiseNum): 
	randX=random.randint(0,src.shape[0]-1)       
	randY=random.randint(0,src.shape[1]-1) 
	 
	if random.random()<=0.5:           
		img[randX,randY]=0       
	else:            
		img[randX,randY]=255    
    return img

#调用函数实现噪声：
noise_gs_img=util.random_noise(img,mode='poisson')
