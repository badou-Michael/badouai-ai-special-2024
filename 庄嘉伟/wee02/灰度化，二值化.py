import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#灰度化
img = cv2.imread("lenna.png")  #加载图片，opencv读取的是BGR格式
h,w = img.shape[:2]  #获取图片的high和wide

#手动处理
img_gray = np.zeros([h,w],img.dtype) #创建一张和原图一样大小的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] =int(m[0]*0.11+m[1]*0.59+m[2]*0.3) #通用格式为RGB,因此转化的时候要注意格式

# print(m)
print(img_gray)
print("灰度化图：%s"%img_gray)
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKeyEx(0)  # 窗口按任意键关闭
    cv2.destroyWindow()  # 销毁窗口指令
# cv_show("gray",img_gray)

plt.subplot(221)  #打印彩图
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

#调用接口灰度化，并打印
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')#把这个数据画到plt上，其中cmap是可选参数，指定颜色映射
print("---image gray----")
print(img_gray)

#二值化
img_binary = np.where(img_gray>=0.5,1,0)
print("---image binary---")
print(img_binary)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')

plt.show()#展示当前绘制图案
