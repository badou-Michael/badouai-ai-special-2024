from PIL import Image, ImageFile
import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage.color import rgb2gray

# 读取图片文件
img = cv2.imread("lenna.png")
print('img.dtype ==%s' % img.dtype)
print('---img-----',img)
h, w = img.shape[:2]  ##获取图片的形状信息 (高度,宽度,通道数)
img_gray = np.zeros([h, w], img.dtype)  # 创建一个和原图像大小一致数组

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  ##进行灰度化

# 图片进行灰度化
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray image',img_gray)##显示图片 在程序执行完毕后自动关闭

# cv2.imwrite('gray_image.png',img_gray)#保存图片 保存在统计目录下

# 将原始图片保存在一个2x2的矩阵中 放在第一个位置上
plt.subplot(221)
img = plt.imread('lenna.png') #该方法获取一个dtype为float32的图片
print('--------img-lenna img dtype----',img.dtype)
plt.imshow(img) #创建图形显示图片

# 将灰度化后的图片放在第二个位置上
# plt.subplot(222)
# plt.imshow(img_gray,cmap='gray')
# print("---image gray----")
# print(img_gray)

#灰度化图片
#将原图的数据类型进行计算 返回相同的数据类型
img_gray= rgb2gray(img)
print('--img_gray-----')
print(img_gray)

#使用openCv灰度化图片
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(222) #放去第二个位置
print('--img_gray-----')
print(img_gray)
plt.imshow(img_gray, cmap='gray')


#使用openCv进行二值化处理
#ret, img_binary = cv2.threshold((img_gray*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )


#二值化图片
#将灰度化后的图片进行二值化处理  大于0.5的像素点为1(白) 小于0.5的像素点为0(黑色)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print('img_binary=====:', img_binary)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
cv2.imwrite('img_binary.png', img_binary)

# 显示图片
plt.show()

# 等待睡眠时间 0为无限制
k = cv2.waitKey(0)
# 判断按键是否是q 为q时推出cv窗口
if k & 0xFF == ord('q'):
    cv2.destroyAllWindows()
