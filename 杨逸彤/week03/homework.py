import cv2
import numpy as np

#1.实现最临近插值和双线性插值
#最临近插值
from matplotlib import pyplot as plt


def function_near(img):
    h,w,c = img.shape
    emptyImg = np.zeros((800,800,c),np.uint8)
    #原图最高最宽是h和w，新图最高最宽都是800，用新/旧求比例可以算出矩阵每一个元素放大的比例
    rateH = 800/h
    rateW = 800/w
    for i in range(800):
        for j in range(800):
            x = int(i/rateH + 0.5)
            y = int(j/rateW + 0.5)
            emptyImg[i,j] = img[x,y]
    return emptyImg

#双线性插值
def function_bilinear(img,out_dim):
    src_h,src_w,c = img.shape
    dst_h,dst_w = out_dim[0], out_dim[1]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,c),np.uint8)
    scale_x, scale_y = float(src_h)/dst_h, float(src_w)/dst_w
    for i in range(c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #计算原点坐标
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x = (dst_x + 0.5) * scale_x - 0.5

                #计算原点坐标向下取整后获取坐标四个邻近点的位置
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                #根据四个点的双线性插值计算目标图像的像素值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = cv2.imread("d:\\Users\ls008\Desktop\lenna.png")
nearImg = function_near(img)
biliImg = function_bilinear(img,(800,800))
cv2.imshow("nearImg",nearImg)
cv2.imshow("biliImg",biliImg)
cv2.imshow("img",img)
cv2.waitKey(0)

#2.证明中心重合+0.5
#设原图坐标（xm,ym),目标图坐标(xn,yn)
#原图中心（xm/2,ym/2),目标图中心(xn/2,yn/2)
#(m-1)/2+z = ((n-1)/2+z)(m/n)
#z-z(m/n) = (n-1)/2 * (m/n) -(m-1)/2
#(1-(m/n))*z = m(n-1)/2n - n(m-1)/2n
#((n-m)/n)*z = (n+m)/2n
#z = (n+m)/2n * n/(n-m)
#z = 1/2

#3.实现直方图均衡化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

#计算给定图像（此处为dst）的直方图。
# 参数中 [dst] 是输入图像，[0] 指定通道（灰度图的唯一通道），
# None 表示不使用掩码，[256] 指定直方图的大小（灰度级数），
# [0, 256] 定义了灰度值的范围
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
#plt.hist(): 绘制直方图。dst.ravel() 将图像数据展平为一维数组，256 指定了直方图的柱子数量。
plt.hist(dst.ravel(), 256)
plt.show()

#在一个窗口中显示图像，此处显示的是原始灰度图像gray和均衡化后的图像dst并排。
#np.hstack(): 将两个数组（图像）水平堆叠，形成一个新图像，用于并排显示。
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
