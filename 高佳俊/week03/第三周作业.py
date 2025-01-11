#  双线性插值
import numpy as np
import cv2

# Bilinear interpolation 是一种在二维网格上进行插值的方法，常用于图像缩放和图像处理。它通过在四个邻近的已知点之间进行线性插值来估算一个未知点的值。
def bilinear_interpolation(img,out_dim):   # 目标图像的尺寸，通常是一个包含目标宽度和高度的元组 (width, height)
    src_h,src_w,channel=img.shape    # 分别表示图像的高度（行数）、宽度（列数）和通道数
    dst_h,dst_w=out_dim[1],out_dim[0]
    print(src_h,src_w)
    print(dst_h,dst_w)
    if src_w==dst_w and src_h==dst_h:
        return img.copy()   # 创建源图像的一个副本，确保原图像在后续操作中不会被意外修改
    dst_img=np.zeros([dst_h,dst_w,3],np.uint8)  # 在某些情况下，uint8 可能被用作一个简短的标记，但 Python 本身不直接提供 uint8 类型  须写成np.uint8
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h   #  计算缩放比例，  使用 float 确保计算结果是浮点数，而不是整数
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 找到目标图像像素点在原图像的位置
                src_x=(dst_x+0.5)*scale_x-0.5
                src_y=(dst_y+0.5)*scale_y-0.5

                # 为了进行双线性插值，找到 源图像中与 src_x 和 src_y 最接近的四个像素点
                src_x0=int(np.floor(src_x))   # np.floor 的结果是浮点数,所以使用int取整
                src_x1=min(src_x0+1,src_w-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,src_h-1)

                # 计算插值
                temp0=img[src_y0,src_x0,i]*(src_x1-src_x)+img[src_y0,src_x1,i]*(src_x-src_x0)   # 这里不需要int：是中间计算结果，这些值通常是浮点数，可以更精确地表示计算结果
                temp1=img[src_y1,src_x0,i]*(src_x1-src_x)+img[src_y1,src_x1,i]*(src_x-src_x0)   # 这里不需要int：是中间计算结果，这些值通常是浮点数，可以更精确地表示计算结果
                dst_img[dst_y,dst_x,i]=int((src_y1-src_y)*temp0+(src_y-src_y0)*temp1)  # 这边的值是最终结果，必须存储在图像数组中，而图像像素通常是整数值，所以需要int

    return dst_img

if __name__=="__main__":
    img=cv2.imread("D:/badou/week02/lenna.png")
    dst=bilinear_interpolation(img,(700,700))
    cv2.imshow("shuangchazhi",dst)
    cv2.waitKey(0)



# 最邻近插值
import numpy as np
import cv2

def zuilinjin(img2):
    height,width,channels=img2.shape
    dst2=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)
            y=int(j/sh+0.5)     # int()是将一个数值转换为整数类型,会对浮点数进行截断，也就是说它会移除小数部分，保留整数部分
            dst2[i,j]=img2[x,y]
    return dst2
img2=cv2.imread("D:/badou/week02/lenna.png")
zoom=zuilinjin(img2)
cv2.imshow("nearest",zoom)
cv2.imshow("img",img2)
cv2.waitKey(0)


import numpy as np
import cv2
import matplotlib.pyplot as plt

# 获取灰度图像
img=cv2.imread("D:/badou/week02/lenna.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",img_gray)
cv2.waitKey(0)

# 直方图法一
plt.figure()
# ravel() 是一个 NumPy 方法，它将二维数组展平为一维数组。在这里，它将整个灰度图像的像素值展平为一个一维数组，以便用于绘制直方图。
# plt.hist()用于计算并绘制数据的直方图。直方图展示了数据的频率分布，即数据值的出现频次。
# 256这是直方图的 bins 参数,指定了直方图的柱子数量。在灰度图像中，灰度值通常在 0 到 255 之间，因此 256 个 bins 对应于每个可能的灰度值。
plt.hist(img_gray.ravel(), 256)   # plt.hist用于绘制直方图
plt.xlabel("xsz")  # 设置 x 轴标签
plt.ylabel("gs")   # 设置 y 轴标签
plt.title("zft")   # 设置图表标题
plt.show()


# 直方图法二
hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")            # X轴标签
plt.ylabel("# of Pixels")     # Y轴标签
plt.plot(hist)                # 用于绘制图形
plt.xlim([0,256])             # 设置 x 轴范围与直方图的 bin 数量一致
plt.show()


# 彩色图像直方图
img2=cv2.imread("D:/badou/week02/lenna.png")
chans=cv2.split(img2)   # 用于将多通道图像分解为其各个单独的通道
colors=("b","g","r")
# for i in chans:
#     cv2.imshow('Blue Channel', i)
#     cv2.waitKey(0)
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for chan,color in zip(chans,colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()


# 直方图均衡化

img3=cv2.imread("D:/badou/week02/lenna.png",0)
# cv2.imshow("grey",img3)
# cv2.waitKey(0)

# 灰度图像直方图均衡化
dst2 = cv2.equalizeHist(img3)

# 计算直方图
plt.subplot(1,2,1)
plt.hist(dst2.ravel(),256)
# plt.show()
plt.subplot(1,2,2)
hist=cv2.calcHist([dst2],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()
cv2.imshow("dbt",np.hstack([img3,dst2]))
cv2.waitKey(0)


# 彩色图像直方图均衡化
img4 = cv2.imread("D:/badou/week02/lenna.png")
cv2.imshow("src", img4)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img4)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
