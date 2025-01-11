import cv2 as cv
import numpy as np

# import matplotlib.pyplot as plt
# import skimage.color as sc  # 可直接调用 rgb2gray
# 1 当前图像lenna 的最邻近插值方法缩放图片;根据原理直接写
image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
# print(image.shape)
cv.imshow('a', image)
height, width, channels = image.shape
near_image = np.zeros((400, 600, channels), dtype=np.uint8)
sh = 400 / height
sw = 600 / width
for i in range(400):
    for j in range(600):
        x = round(i / sh)
        y = round(j / sw)
        near_image[i, j] = image[x, y]
print(near_image.shape)
print(near_image.size)
cv.imshow('near_image', near_image)


# 2通过自定义函数  最邻近插值法
def nearest_image(image):
    height, width, channels = image.shape
    near_image = np.zeros((600, 600, channels), dtype=np.uint8)
    sh = 600 / height
    sw = 600 / width
    for i in range(600):
        for j in range(600):
            x = int(i / sh + 0.5)
            # 将 目标图像的坐标h 除以 缩放比 四舍五入(因int取整是向下的,故比值+0.5处理以实现结果的四舍五入),去找邻近的原图像的坐标 ,同理去找邻近的y坐标;
            y = int(j / sw + 0.5)
            near_image[i, j] = image[x, y]
    return near_image


image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
print(image.shape)
near_image1 = nearest_image(image)
print(near_image1)
cv.imshow('near_image1', near_image1)
cv.imshow('image', image)
cv.waitKey(0)

# 3 通过接口直接调用  最邻近插值法
image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
h = int(input("请设置目标图像的高:"))
w = int(input("请设置目标图像的宽:"))
# c = image.shape[2]
near_image = cv.resize(image, (h, w))
cv.imshow('image', image)
# near_image =  cv.resize(img, (800,800,c),near)显示,不存在near ;删除之后也报错说传参太多了,咨询老师待复
cv.imshow('near_image', near_image)

# # 2.1 双线性插值法: 按原理
import cv2 as cv
import numpy as np
from numpy.core._multiarray_umath import ndarray


def bilinear(image, out_dim):
    src_h, src_w, channels = image.shape  # 定义src_h,src_w分别是原图像(输入图像)的高,宽
    dst_h, dst_w = out_dim[1], out_dim[0]  # 定义src_h,src_w分别是目标图像(想要输出图像)的高,宽
    print('src_h,src_w =', src_h, src_w)
    print('dst_h,dst_w =', dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return image.copy()  # 待尝试 直接返回 image 是否可以??
    dst_image = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 创建目标图像大小的全零数组
    # 计算原图src与目标图dst的宽(即X方向),高(即y方向)缩放比
    scale_x, scale_y = float(src_w / dst_w), float(src_h / dst_h)
    for i in range(3):  # 遍历通道数
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):  # 遍历 目标图dst的行列位置(即各像素点的坐标dst_x,dst_y)
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 通过目标dst 像素点坐标 去找 对应在原图src中的坐标src_x,src_y
                # find the coordinates of the points which will be used to compute the interpolation
                # 译:找到用来计算插值的点的坐标,想一下 示例图中的四个点:Q11(X1,Y1),Q21(X2,Y1),Q12(X1,Y2),Q22(X2,Y2)
                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, src_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_h - 1)
                # calculate the interpolation
                # 译:计算插值
                tmp_r1 = (src_x2 - src_x) * image[src_y1, src_x1, i] + (src_x - src_x1) * image[src_y1, src_x2, i]
                tmp_r2 = (src_x2 - src_x) * image[src_y2, src_x1, i] + (src_x - src_x1) * image[src_y2, src_x2, i]
                dst_image[dst_y, dst_x, i] = int((src_y2 - src_y) * tmp_r1 + (src_y - src_y1) * tmp_r2)

    return dst_image

if __name__ == "__main__":
    image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
    dst = bilinear(image, (600, 600))
    cv.imshow('image', image)
    cv.imshow('bilinear_image', dst)
    cv.waitKey(0)

## 双线性插值 直接调用接口

import numpy as np
import cv2 as cv
image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
image_bilinear = cv.resize(image,(600,600),interpolation=cv.INTER_LINEAR)
cv.imshow('image',image)
cv.imshow('image_bilinear',image_bilinear)


## 3.直方图均值化
# 3.1 灰度图直方图均衡化
# 获取灰度图像
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png", 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv.equalizeHist(gray)      # 调用接口,直接使灰度图gray 的直方图均衡化赋值给dst
cv.imshow('a',img)
cv.imshow('b',gray)
cv.imshow('c',dst)
cv.waitKey(0)


# 3.2 彩色图像直方图均衡化
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("E:/GUO_APP/GUO_AI/picture/lenna.png", 1)
cv.imshow("a", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化

(b, g, r) = cv.split(img)     # 利用split 函数分离通道
# 分别对每个通道进行直方图均衡化
bH = cv.equalizeHist(b)
gH = cv.equalizeHist(g)
rH = cv.equalizeHist(r)
# 合并每一个通道
dst = cv.merge((bH, gH, rH))   # 利用merge函数合并通道
cv.imshow("b", dst)
cv.waitKey(0)


