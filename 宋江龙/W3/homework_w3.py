# -*- coding:utf-8 -*-
# @Time:2024/9/11 20:36

import cv2
import numpy as np

'''
最临近插值法一：不使用已经封装好的函数，自定义函数和缩放尺寸，
自定义函数步骤：
1，传参：图片，预计缩放的尺寸高和宽
2，获取所传图片的高、宽和通道数
3，定义空图片并计算缩放比例，用目标图的高和宽，分别除原图的高和宽
4，先高后宽进行遍历空白图，并将缩放比例还原后获取原图位置的像素点同时赋给空白图
'''

def nearestInterpolationOne(image, high, wide):
    image_array = cv2.imread(image)
    src_high, src_wide, channels = image_array.shape
    if src_high==high and src_wide==wide:
        return image_array.copy()
    emptyImage = np.zeros((high, wide, channels), np.uint8)
    sh, sw = high/src_high, wide/src_wide
    for i in range(high):
        for j in range(wide):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i,j] = image_array[x,y]
    return emptyImage


'''
最临近插值法二：使用cv2.resize函数，interpolation参数值为0，输入图片和缩放尺寸即可
resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
参数	   解释
src	       输入原图像
dsize	   输出图像的大小，方式：(宽,高)
fx	       width方向的缩放比例
fy	       height方向的缩放比例
interpolation	插值方式，默认为1：双线性插值，0为最临近插值，还有2、3、4三种方式
'''
def nearestInterpolationTwo(image, high, wide):
    img_array = cv2.imread(image)
    new_image = cv2.resize(img_array,(high, wide),interpolation=0)
    return new_image


'''
双线性插值法一：不使用已经封装好的函数，自定义函数，并自定义缩放尺寸
自定义函数步骤：
1，传参：图片，预计缩放的尺寸高和宽
2，获取所传图片的高、宽和通道数
3，定义空图片并计算缩放比例，用“原图”的高和宽，分别除“目标图”的高和宽（这是和最临近插值不一样的地方）
4，通过比例关系计算出原图的像素位置：原图宽x=目标图宽x*(缩放比例：原图宽/目标图宽），原图高y=目标图y*(缩放比例：原图高/目标图高），
公式中的0.5是为了原图和目标图的中心对齐
5，遍历时增加一层通道的遍历，因为不是直接赋值，需要通过计算得到目标像素值
'''
def bilinearInterpolationOne(image, high, wide):
    image_array = cv2.imread(image)
    src_high, src_wide, channels = image_array.shape
    if src_high==high and src_wide==wide:
        return image_array.copy()
    empty_image = np.zeros((high, wide, channels), dtype=np.uint8)
    sh, sw = float(src_high) / high, float(src_wide) / wide
    for i in range(channels):
        for dst_x in range(high):
            for dst_y in range(wide):
                # 确认两个图的中心，然后按比例关系计算出原图的像素位置
                src_x = (dst_x + 0.5) * sh - 0.5
                src_y = (dst_y + 0.5) * sw - 0.5
                # 找出可以用来计算目标图像素值的4个原图坐标（X0,Y0),(X0,Y1),(X1,Y0),(X1,Y1),X0和X1是相邻点，距离是1，Y同理
                x0 = int(np.floor(src_x))
                # 为防止越界，在加1后的数据和总宽度减一的数据中取最小值
                x1 = min(x0 + 1, src_wide - 1)
                y0 = int(np.floor(src_y))
                y1 = min(y0 + 1, src_high - 1)
                # 将数据带入公式，计算出两个临时中间点的位置
                template0 = (x1 - src_x) * image_array[x0, y0, i] + (src_x - x0) * image_array[x1, y0, i]
                template1 = (x1 - src_x) * image_array[x0, y1, i] + (src_x - x0) * image_array[x1, y1, i]
                # 根据中间点位置和公式计算出目标点的位置，并赋予像素值
                empty_image[dst_x, dst_y, i] = int((y1 - src_y) * template0 + (src_y - y0) * template1)
    return empty_image

'''双线性插值法二：使用cv2.resize函数，interpolation参数值为1，输入图片和缩放尺寸即可'''
def bilinearInterpolationTwo(image, high, wide):
    img_array = cv2.imread(image)
    new_image = cv2.resize(img_array,(high, wide),interpolation=1)
    return new_image

'''彩色图像直方图均衡化
1，读取图片，然后用cv2.split函数将图片分解成单独的颜色通道
2，使用equalizeHist函数分别将各个颜色通道的直方图均衡化
3，最后进行合并生成均衡化的彩色图像'''
def colorImageEqualize(image):
    image_array = cv2.imread(image)
    (b, g, r) = cv2.split(image_array)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    result = cv2.merge((bh, gh, rh))
    return result



if __name__=='__main__':
    image = "D:\exercise_sjlong\my_project\data\lenna.png"
    # zoom_one = nearestInterpolationOne(image, 400, 600)
    # zoom_two = nearestInterpolationTwo(image, 400, 600)
    # zoom_three = bilinearInterpolationOne(image, 200, 400)
    # zoom_four = bilinearInterpolationTwo(image, 400, 200)
    result = colorImageEqualize(image)
    # cv2.imshow("nearest interp one",zoom_one)
    # cv2.imshow("nearest interp two",zoom_two)
    # cv2.imshow("nearest interp three",zoom_three)
    # cv2.imshow("nearest interp four",zoom_four)
    cv2.imshow("new", result)
    cv2.waitKey(0)







