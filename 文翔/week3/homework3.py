import cv2
import numpy as np
from matplotlib import pyplot as plt

# 插值算法的本质：求出虚拟点上的像素值
# 实现最临近插值和双线性插值
# 最邻近实现思路：虚拟点的像素值取距离最近的点
def nearest_interp(img):
    high,width,channelS = img.shape
    emptyImg = np.zeros((800,800,channelS),np.uint8)
    for i in range(800):
        for j in range(800):
            emptyImg[i,j] = img[int(high*i/800+0.5),int(width*j/800+0.5)]
    return emptyImg

# img=cv2.imread("lenna.png")
# zoom=nearest_interp(img)
# print(zoom)
# print(zoom.shape)
# cv2.imshow("nearest interp",zoom)
# cv2.imshow("image",img)
# cv2.waitKey(0)

def bilinear_interp(img,out_dim):
    high,width,channels = img.shape
    dst_y,dst_x = out_dim[0],out_dim[1]
    src_y,src_x = high,width
    dst_img = np.zeros((dst_y,dst_x,channels),np.uint8)
    high_rate = src_y / dst_y
    width_rate = src_x / dst_x
    for i in range(dst_y):
        for j in range(dst_x):
            srcY = (i+0.5)*high_rate - 0.5
            srcX = (j+0.5)*width_rate - 0.5
           # 使得变换前后两图中心对齐

            y0 = int(srcY)  #向下取整得第一个数据点
            x0 = int(srcX)
            y1 = min(y0+1,src_y-1)  #第二个数据点，注意限制其边界(不超过原图)
            x1 = min(x0+1,src_x-1)

            #print(x0,y0,x1,y1)
            #得到两个数据点，下面进行代入双线性插值公式，注：是要在点(i,j)处插值，如何计算此处的像素值
            # 注：为保证后期不除以0，默认x1-x0 = 1
            # temp0 = (x1-j)/(x1-x0)*img[x0,y0] + (j-x0)/(x1-x0)*img[x1,y0]
            # temp1 = (x1-j)/(x1-x0)*img[x0,y1] + (j-x0)/(x1-x0)*img[x1,y1]
            temp0 = (x1-srcX)*img[y0,x0] + (srcX-x0)*img[y0,x1]
            temp1 = (x1-srcX)*img[y1,x0] + (srcX-x0)*img[y1,x1]
            dst_img[i,j] = (y1-srcY)*temp0 + (srcY-y0)*temp1
    return dst_img

# img = cv2.imread('lenna.png')
# dst = bilinear_interp(img,(700,700))
# cv2.imshow('bilinear interp',dst)
# cv2.waitKey()

# 获取灰度图像
img = cv2.imread("lenna.png", 1)  #1为彩色、0为灰色
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)







