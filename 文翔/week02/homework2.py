#图像灰度化和二值化的实现

# 知识点：每一个图像都可以看作一个个像素组成
# 灰度图：每个像素由0-255表示，0为黑色，255为白色
# 二值图：每个像素非黑即白，只有0和255两个值
# RGB图：每个像素由一个三维坐标表示，红绿蓝三原色（red,green,blue）
#       对应维度的坐标值表示该成分颜色的占比，值越大越明显，（255，0，0）为红色，以此类推

# 不同图之间的转换：以RGB->灰度图->二值图为例

# RGB->灰度图 Gray = 0.3R + 0.59G + 0.11B
# 灰度值->二值图 设置阈值，一般先将gray归一化，然后以0.5为阈值

from skimage.color import rgb2gray
import numpy as np
#import matplotlib.pyplot as plt  暂时不用，用cv2画图
from PIL import Image
import cv2

# RGB->灰度图 Gray = 0.3R + 0.59G + 0.11B
# 思路：读入图片，按行列获取三维数组，然后设立相同行列的数组，按公式依次计算即可
img = cv2.imread("lenna.png")
#print(f'img的值是{img.shape}')
cols,rows = img.shape[:2] #img为numpy对象，shape查看维度，[:2]表示切片（取前两个），即从第一个开始，不包括2
img_gray = np.zeros([cols,rows],img.dtype) #创建同样维数的空数组作为灰度图
#方法一：按公式计算
for i in range(cols):
    for j in range(rows):
        img_gray[i,j] = 0.3*img[i,j,2] + 0.59*img[i,j,1] + 0.11*img[i,j,0]

#方法二：直接调用库函数
# img_gray = rgb2gray(img)

cv2.imshow("img_gray",img_gray)
cv2.imshow("img",img)

# 灰度值->二值图 设置阈值，一般先将gray归一化，然后以0.5为阈值
img_norm = img_gray.astype('float32') / 255.0

#方法一：where语句，注意cv2的imshow不支持float32，下面进行了转换
img_binary_float32 = np.where(img_norm >= 0.5, 1, 0)
img_binary = (img_binary_float32 * 255).astype(np.uint8)

#方法二，定义
# img_binary = np.zeros([cols,rows],img.dtype)
# for i in range(cols):
#     for j in range(rows):
#         if img_norm[i,j] >=0.5:
#             img_binary[i,j] = 255
#         else:
#             img_binary[i,j] = 0
cv2.imshow("img_binary",img_binary)

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
