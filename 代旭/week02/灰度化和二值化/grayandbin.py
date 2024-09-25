from skimage.color import  rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#读取图片 【注意】opencv输出img是BGR 不是 RGB
img = cv2.imread("lenna.png")
#获取图片的height width
h,w=img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
 for j in range(w):
  m = img[i,j] #取出当前h和w中的BGR坐标
  img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3) #将BGR坐标转化为gray坐标并赋值给新图像

print(m)
print(img_gray)
print("image show gray:%s"%img_gray)
#cv2.imshow("image show gray",img_gray)

plt.subplot(221)
img = plt.imread("lenna.png") #原图
plt.imshow(img)

#灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

#二值化

img_binary = np.where(img_gray>=0.5,1,0)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')

rows,cols = img_gray.shape
for i in range(rows):
 for j in range(cols):
  if(img_gray[i,j]<=0.5):
   img_gray[i,j]=0
  else:
   img_gray[i,j]=1

plt.subplot(224)
plt.imshow(img_gray,cmap='gray')
plt.show()
