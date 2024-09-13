import cv2
import numpy as np
def equalized(image):#直方图均衡化
    h,w=image.shape
    count,gray=[0]*256,0
    emptyImage=np.zeros((h,w),np.uint8)
    sum=0
    for i in range(h):
        for j in range(w):
            gray=image[i,j]
            count[gray]+=1 #循环统计
    for i in range(h):
        for j in range(w):
            for k in range(image[i,j]):
              sum+=count[k]   #计算像素总数
            q=sum/(h*w)*255-1 #计算像素值
            emptyImage[i,j]=q
            sum=0
    return emptyImage
image=cv2.imread('lenna.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#equalized_gray=cv2.equalizeHist(gray) #直接使用cv2内接口
equalized_gray=equalized(gray)         #编写函数
# 转换到HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 对亮度通道进行直方图均衡化
hsv_image[:, :, 2] = equalized(hsv_image[:, :, 2])
#hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
hsv_image=cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)
imgs_gray=np.hstack([gray,equalized_gray])
imgs_color=np.hstack([image,hsv_image])
cv2.imshow('gray',imgs_gray)
cv2.imshow('color',imgs_color)
cv2.waitKey()
