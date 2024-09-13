"""
@author:张扣来
彩色图像灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image
import cv2
"""
灰度化1：浮点算法
"""
# 读取一张图片
img = cv2.imread("../../request/task2/lenna.png")
# 获取图片的height和width
h,w = img.shape[:2]
"""
1）创建一张和当前图片一样大小的单通道图片,并定义灰度图片数据类型
2）img.dtype默认输出uint8，说明每个数组元素用一个字节(8位)保存，
3)每个数组元素为一个像素的B、G、R通道的颜色值，颜色值取值范围为[0,255]
"""
img_gray = np.zeros([h,w],img.dtype)
# 将原图的BGR数据转化为灰度值数据
for i in range(h):
    for j in range(w):
        m = img[i,j]
        # 换算灰度值bgr坐标值
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
# 输出m、img_gray数据
print (m)
print(img_gray)
print("image show gray:%s" % img_gray)
# 在名称为image show gray的窗口中显示出来
cv2.imshow("image show gray",img_gray)

"""
灰度化2：RGB分量加权算法
"""
# 在matplotb中的一个2x2的子图网格中创建一个子图，这个子图位于网格的左上角第1个位置
plt.subplot(331)
# 读取图片
img = plt.imread("../../request/task2/lenna.png")
# 显示图像
plt.imshow(img)
print("------img lena------")
print(img)
# 进行灰度化处理，将真彩色的img数值数据转换成灰度值图像数据，此时转换的数据类型为double类型
img_gray = rgb2gray(img)
# 激活2x2网格中的第2个位置
plt.subplot(332)
# 将221中的处理后的imag_gray灰度数据映射到222中灰度的颜色图谱中，camp有以下三个参数值，jet彩虹渐变，viridis为绿色
plt.imshow(img_gray,cmap="jet")
plt.subplot(333)
plt.imshow(img_gray,cmap="viridis")
plt.subplot(334)
plt.imshow(img_gray,cmap="gray")
plt.show()
print("------img_gray------")
print(img_gray)

"""
二值化
"""
# img_gray中数值大于0.5转为1，否则转为0，输出二值化
img_binary = np.where(img_gray >=0.5,1,0)
print("------img_binary------")
print(img_binary)
print("------img_shape------")
# 输出形状长宽
print(img_binary.shape)
plt.subplot(111)
plt.imshow(img_binary,cmap="gray")
plt.show()

