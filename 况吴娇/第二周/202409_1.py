##author : wujiao kuang
##202409
##coding: utf-8
print('彩色图像灰度化、二值化  BY KWJ')

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color  import rgb2gray
from PIL import Image

##############cv2   灰度
img_path= r"lenna.png"
img=cv2.imread(img_path)
##Open CV默认为bGr，直接用颜色不正常
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow("lena gray",img_gray)
# cv2.waitKey(0)##毫秒数作为参数
# cv2.destroyAllWindows()###关闭有Cpen CV创建的窗口
h,w=img.shape[:2]#获取高和宽
img_gray=np.zeros([h,w],img.dtype)###高，宽，创建和当前图片大小一样的单通道,返回的3个数
for i in range(h):
    for j in range (w):
        m=img[i,j]##3个数
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)##浮点算法：Gray=R*0.3+G*0.59+L*0.11L

print(m)
print(img_gray)
print("img gray %s" %img_gray)
cv2.imshow("img_gray",img_gray)

#cv2.waitKey(1000)##1秒
##cv2.imwrite(r"F:\BADOU\bin\python\lena.png",img_gray)  #保存的路径要写完整


#########画布
# matplotlib
##1
plt.subplot(2,2,1)
img=plt.imread(img_path)
plt.imshow(img)
#灰度  skimage
##2
img_gray=rgb2gray(img)
plt.subplot(2,2,2)
plt.imshow(img_gray,cmap="gray")
#####二值化##通常首先将彩色图片转化为灰度图像
#方法1
##np
plt.subplot(2,2,3)
# img_binary=np.where(img_gray>=0.5,1,0)
# plt.imshow(img_binary,cmap="gray")
# print("img_binary:%s" %img_binary)
# print("img_binary.shape:%s" %img_binary.shape)
#方法2
rows,cols =img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j]>=0.5 :
            img_gray[i, j]=1
        else:
            img_gray[i, j]=0
plt.imshow(img_gray,cmap="gray")

##保存整个图像窗口
##plt.savefig("lenna_processed.png")

###保存二值图
##plt.imsave 直接从图像数据数组保存图像，而非matplotlib的图像对象保存，所以无需绘制；
#plt.imsave('lenna_save.png',(img_gray*5).astype(np.uint8),cmap="gray")
##保存图象时，二值图像的值乘以225，并转化为uint8，会保持清晰的对比度，并在标准的图像显示器中正确显示
plt.show()#阻塞直到窗口关闭
