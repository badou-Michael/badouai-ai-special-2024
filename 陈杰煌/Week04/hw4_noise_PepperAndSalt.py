# 椒盐噪声(salt and pepper noise)
# 椒盐噪声又称为脉冲噪声，它是一种随机出现的白点或者黑点
# • 椒盐噪声 = **椒噪声 （pepper noise）+ 盐噪声（salt noise）**。 椒盐噪声的值为0(椒)或者255(盐)。
# 前者是低灰度噪声，后者属于高灰度噪声。一般两种噪声同时出现，呈现在图像上就是黑白杂点。
# 对于彩色图像，也有可能表现为在单个像素BGR三个通道随机出现的255或0。
# 如果通信时出错，部分像素的值在传输时丢失，就会发生这种噪声。

import numpy as np
import cv2
import random
import time

def pepper_and_salt_noise(src, percetage):
    noiseImg = src.copy()  # 使用副本以避免修改原始图像
    noiseNum = int(percetage * src.shape[0] * src.shape[1])
    
    # 生成所有可能的点并随机选择 noiseNum 个点
    all_points = [(x, y) for x in range(1, src.shape[0] - 1) for y in range(1, src.shape[1] - 1)]
    selected_points = random.sample(all_points, noiseNum)
    
    # 随机使选中的点变成0或255
    for (randX, randY) in selected_points:
        noiseImg[randX, randY] = 0 if random.random() <= 0.5 else 255
    
    return noiseImg

# 读取灰度图像：模式0表示灰度图像
src_img_gray = cv2.imread('lenna.png', 0)

# 参数
percentage = 0.2

# add noise
start_time = time.time()  # 记录开始时间
img1 = pepper_and_salt_noise(src_img_gray, percentage)
end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算执行时间
print(f"pepper_and_salt_noise 函数执行时间: {execution_time:.6f} 秒")

# 显示图像
cv2.imshow('Source', src_img_gray)
cv2.imshow('lenna_PepperAndSaltNoise', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()