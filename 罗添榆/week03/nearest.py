import cv2
import numpy as np

#最邻近插值
def nearest(img):
    #先获取图像的高度，宽度和通道数
    height,width,channels=img.shape
    #再创建一个空的目标图（调整到多少像素）
    emptyImage=np.zeros((800,800,channels),np.uint8)
    #缩放比例
    sh=800/height
    sw=800/width
    #最邻近插值实现
    #从0到800开始遍历
    for i in range(800):
        for j in range(800):
            #对应点在原图的位置，进行向下取整，+0.5是保证可以四舍五入
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            #将向下取整后的点赋值给空的目标图
            emptyImage[i,j]=img[x,y]
    return emptyImage

#读取原图
img = cv2.imread("lenna.png")
#最邻近插值处理
zoom = nearest(img)
print(zoom)
print(zoom.shape)
#显示原图与目标图，进行对比
cv2.imshow("Nearest Interp",zoom)
cv2.imshow("Img",img)
#CV2的imshow如果要一直显示需要调用waitkey函数
cv2.waitKey(0)
