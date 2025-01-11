# 高斯噪声(Gaussian noise)
# 概率密度函数服从高斯分布的一类噪声
# 通常用于模拟电子元件的热噪声，如电阻、晶体管、二极管等
# 也可以用于模拟信道的噪声，如无线通信、有线通信等
# 还可以用于模拟图像、音频、视频和文本的噪声

#一个正常的高斯采样分布公式, 得到输出像素P_out `P_out = P_in + random.gauss`

# 其中random.gauss是通过sigma和mean来生成符合高斯分布的随机数。


import numpy as np
import cv2
from numpy import shape
import random
import time 


def GaussianNoise(src_pic, means, sigma, percetage):
    noiseImg = src_pic.copy()  # 使用副本以避免修改原始图像
    noiseNum = int(percetage * src_pic.shape[0] * src_pic.shape[1])
    
    """
    for i in range(noiseNum):
        #每次取一个随机点
		#把一张图片的像素用行和列表示的话, randX 代表随机生成的行, randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX = random.randint(1,src_pic.shape[0]-1)    #randint(a,b) 随机生成a<=N<=b的整数, 内置种子不重复
        randY = random.randint(1,src_pic.shape[1]-1)
    """
    # 生成所有可能的点
    all_points = [(x, y) for x in range(1, src_pic.shape[0] - 1) for y in range(1, src_pic.shape[1] - 1)]
    
    # 随机打乱这些点
    random.shuffle(all_points)
    
    # 选择前 noiseNum 个点
    selected_points = all_points[:noiseNum]
    
    for (randX, randY) in selected_points:
        # 在原有像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
            
    return noiseImg


# read gray image: mdoe 0 for gray
src_img_gray = cv2.imread('lenna.png',0)

# parameters
sigma = 2
means = 4
percentage = 0.8

# add noise
start_time = time.time()  # 记录开始时间
img1 = GaussianNoise(src_img_gray, means, sigma, percentage)
end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算执行时间
print(f"GaussianNoise 函数执行时间: {execution_time:.6f} 秒")

# show images
src_img = cv2.imread('lenna.png')

# convert to gray
img2 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# show images
cv2.imshow('Source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



