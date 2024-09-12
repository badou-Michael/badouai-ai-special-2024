#最临近插值
import cv2
import numpy as np
def function(img):
    height,width,channels = img.shape
    emptyimage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
           x=int(i/sh+0.5)
           y=int(j/sh+0.5)
           emptyimage[i,j]=img[x,y]
    return emptyimage
img=cv2.imread("lenna.png")
zoom=function(img)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

#双线性插值
import cv2
import numpy as np
def  bilinear_interpolation(img,out_dim):
    scr_h,scr_w,channel=img.shape
    dst_h,dst_w=out_dim[1],out_dim[0]
    print("scr_h,scr_w=",scr_h,scr_w)
    print("dst_h,dst_w=",dst_h,dst_w)
    if scr_h==dst_h and scr_w==dst_w:
        return img.copy()
    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y=float(scr_w/dst_w),float(scr_h/dst_h)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                scr_x=(dst_x+0.5)*scale_x-0.5
                scr_y=(dst_y+0.5)*scale_y-0.5
                scr_x0=int(np.floor(scr_x))
                scr_x1=min(scr_x0+1,scr_w-1)
                scr_y0=int(np.floor(scr_y))
                scr_y1=min(scr_x0+1,scr_h-1)

                temp0 = (scr_x1 - scr_x) * img[scr_y0, scr_x0, i] + (scr_x - scr_x0) * img[scr_y0, scr_x1, i]
                temp1 = (scr_x1 - scr_x) * img[scr_y1, scr_x0, i] + (scr_x - scr_x0) * img[scr_y1, scr_x1, i]
                dst_img[dst_y, dst_x, i] = int((scr_y1 - scr_y) * temp0 + (scr_y - scr_y0) * temp1)
    return dst_img

if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    dst=bilinear_interpolation(img,(700,700))
    cv2.imshow("bilinear interp",dst)
    cv2.waitKey(0)

#直方图均衡化
# 灰度图像直方图均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst=cv2.equalizeHist(gray)
hist=cv2.calcHist([dst],[0],None,[256],[0.256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)


#证明中心重合+0.5
原图像M*M   目标图像N*N
目标图像在原图像的坐标值(x,y)
原图像坐标(xm,ym) m=0,1,2...M-1 几何中心（X M-1/2,Y M-1/2)
目标图像坐标(xn,yn) n=0,1,2...M-1 几何中心（X N-1/2,Y N-1/2)
x/n=M/N
M-1/2+Z=(N-1/2+Z)M/N
Z(1-M/N)=(N-1)M/2N-(M-1)N/2N
Z(N-M/N)=N-M/2N
Z=1/2
