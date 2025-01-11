import cv2
import numpy as np

###定义函数
def function (img):
    height,width,channels=img.shape
    print("图像通道数",channels)
    emptyimage=np.zeros((800,800,channels),np.uint8)##uint8代表无符号的8位整数，其取值范围从0到255
    sh = 800 / height
    sw = 800 / width
    for i in range (800):
        for j in range (800):
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            emptyimage[i,j]=img[x,y]
    return emptyimage



img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("lenna",img)
cv2.waitKey(0)

#
# import cv2
#
# resized_img=cv2.resize(cv2.imread("lenna.png"),(800,800),interpolation=cv2.INTER_LINEAR)
# # # 读取图像
# # img = cv2.imread('lenna.png')
# # # 改变图像大小 cv2.resize 是 OpenCV 库中的一个函数，用于改变图像的大小。这个函数非常灵活，允许你指定新的图像尺寸，以及用于计算新像素值的插值方法
# # resized_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
#
# # interpolation：插值方法，这是一个指定如何进行像素插值的参数，常用的插值方法包括：
# # cv2.INTER_NEAREST - 最近邻插值
# # cv2.INTER_LINEAR - 双线性插值（默认值）
# # 显示图像
# cv2.imshow('Resized Image', resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


