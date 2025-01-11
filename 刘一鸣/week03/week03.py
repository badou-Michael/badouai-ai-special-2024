
'''
作业：
1.实现最临近插值和双线性插值；
2.证明中心重合+0.5；
3.实现直方图均衡化。
'''
import cv2
import numpy as np

#双线性插值
def biAdd(img,zoomNum):
    src_h, src_w, channel = img.shape
    dest_h, dest_w = src_h + zoomNum, src_w + zoomNum
    scale_h = src_h / dest_h
    scale_w = src_w / dest_w
    emptyImage = np.zeros((dest_h, dest_w, channel), np.uint8)

    for i in range(dest_h):
        for j in range(dest_w):
            srch = (i+0.5) * scale_h - 0.5
            srcw = (j+0.5) * scale_w - 0.5

            src_heightLow =int(np.floor(srch))
            src_heightHigh =min(src_heightLow+1,src_h-1)
            src_weighLow = int(np.floor(srcw))
            src_weighHigh = min(src_weighLow + 1, src_w - 1)

            temp0=(src_weighHigh-srcw)*img[src_heightLow,src_weighLow]+(srcw-src_weighLow)*img[src_heightLow,src_weighHigh]
            temp1=(src_weighHigh - srcw) * img[src_heightHigh,src_weighLow] + (srcw - src_weighLow) * img[src_heightHigh,src_weighHigh]

            emptyImage[i, j] = (src_heightHigh-srch)*temp0+(srch-src_heightLow)*temp1
    return emptyImage

#最临近插值
def nearest(img,zoomNum):
    src_h, src_w, channel = img.shape
    dest_h, dest_w = src_h + zoomNum, src_w + zoomNum
    scale_h = src_h / dest_h
    scale_w = src_w / dest_w
    emptyImage = np.zeros((dest_h, dest_w, channel), np.uint8)

    for i in range(dest_h):
        for j in range(dest_w):
            srch = int(i * scale_h + 0.5)
            srcw = int(j * scale_w + 0.5)

            emptyImage[i, j] = img[srch, srcw]
    return emptyImage


#主函数
img=cv2.imread('lenna.png')
#1.1实现最临近插值
imgZoom = nearest(img,300)
#1.2双线性插值
imgBiAdd = biAdd(img,300)

cv2.imshow('image',img)
cv2.imshow('imageNearest',imgZoom)
cv2.imshow('imageBi',imgBiAdd)


#3.1直方图均衡化
(b,g,r) =cv2.split(img)
bh=cv2.equalizeHist(b)
gh=cv2.equalizeHist(g)
rh=cv2.equalizeHist(r)
imgBalance=cv2.merge((bh,gh,rh))
cv2.imshow('imgBalance',imgBalance)

cv2.waitKey(0)

#2.1
#每个像素点间距是1，找到中心就是+0.5。
