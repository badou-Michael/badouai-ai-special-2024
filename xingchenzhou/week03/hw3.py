#1最邻近插值
import cv2
import numpy as np
from matplotlib import pyplot as plt

#1最邻近插值和双线性插值
def nearest_interpolation(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)  
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage


def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))     #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img



img = cv2.imread("lenna.png")
p1 = nearest_interpolation(img)
p2 = bilinear_interpolation(img,(700,700))
print(img.shape)
print(p1.shape)
print(p2.shape)

'''
2证明中心重合+0.5
原图像（MxM) 目标图像(NxN)
  目标图像在原图像的坐标位置为(x,y)
  原图坐标(xm,ym) m = 0,1,2...,m-1 几何中心为（xm-1/2,ym-1/2)
  目标图图坐标(xn,yn) n = 0,1,2...,n-1 几何中心为（xn-1/2,yn-1/2)
  x = n(M/N) ----> ((M-1)/2)+z=(((N-1)/2)+Z)*M/N
            -----> z = 1/2
  根据比例缩放公式可知两边+1/2即可几何中心相同         
'''
#3实现直方图均衡化

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
