import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1.1 最临近插值
def function(img):
    height,width,channels=img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage


img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

# cv2.resize(img,(800,800,3),near/bin)

# 1.2 双线性插值
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear inter', dst)
    cv2.waitKey()

# 2.证明中心重合+0.5
原图像M*M,目标图像N*N,目标图像在原图像坐标系位置为（x, y）
原图坐标（Xm,Ym） m=0,1,2,3...M-1  几何中心（X(M-1)/2,Y(M-1)/2）
目标图像坐标（Xn,Yn） n=0,1,2,3...N-1  几何中心（X(N-1)/2,Y(N-1)/2）
若要使几何中心相同，则(M-1)/2=(N-1)/2*(M/N),左右两边添加未知变量Z，则：
(M-1)/2+Z=((N-1)/2+Z)*(M/N)
(M-1+2Z)/2=(N-1+2Z)/2*(M/N)=(MN-M+2ZM)/2N
2NM-2N+4ZN=2MN-2M+4ZM
4ZN-4ZM=2N-2M
4Z(N-M)=2(N-M)
得Z=0.5


# 3.实现直方图均衡化
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
print(dst)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
print(hist)

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
