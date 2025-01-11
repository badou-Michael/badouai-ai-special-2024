import cv2
import numpy as np
# 定义function函数，手写放大缩小的过程
def function(img):
    # 取出图像的高、宽、通道数
    height,width,channels =img.shape
    # 建立空的矩阵（全0，把原图放大到800*800,通道数不变，无符号整数）
    emptyImage=np.zeros((800,800,channels),np.uint8)
    # 计算缩放比例
    sh=800/height
    sw=800/width
    # 遍历像素点
    for i in range(800):
        for j in range(800):
            # i/sh就是换算为原图的坐标，0.5为了避免误差，因为int向下取整，四舍五入
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            # 赋值
            emptyImage[i,j]=img[x,y]
    return emptyImage

# 接口直接实现放大缩小功能
# cv2.resize(img, (800,800,c),near/bin)

# 读取图像
img=cv2.imread("lenna.png")
# 放大缩小，调用function
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


