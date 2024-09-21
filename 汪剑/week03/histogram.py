import numpy as np
import cv2
import matplotlib.pyplot as plt

# 方法一
# 灰度图像直方图
img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 生成灰度图像

plt.figure()

plt.hist(gray_img.ravel(), 256)  # ravel()，将多维数组展成一维数组

# 生成图标标题和坐标标签
plt.title('Gray Histogram')
plt.xlabel('Pixel Intersity')
plt.ylabel('Frequency')

plt.show()



# 彩色图像直方图
img = cv2.imread('lenna.png')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 分离3个通道
r_img, g_img, b_img = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

'''
plt.figure(num,figsize,dpi,facecolor,edgecolor,frameon)函数：
num：整数或字符串，设置图形的编号或名称，默认值：自动递增编号
figsize：tuple(宽度,高度)，单位英寸。控制图形窗口大小，默认值：(6.4,4.8)
dpi：整数，控制图形的分辨率，默认值：100
facecolor：颜色字符串或RGB值，设置图形的背景颜色，默认值：white
edgecolor：颜色字符串或RGB值，设置图形边框的颜色，默认值：white
frameon：True/False，决定图形是否绘制背景框，默认值：True
'''
plt.figure(figsize=(10,8))


# 生成各通道直方图，在同一张图上展示
plt.hist(r_img.ravel(),256,color='red',label='Red Channel')
plt.hist(g_img.ravel(),256,color='green',label='Green Channel')
plt.hist(b_img.ravel(),256,color='blue',label='Blue Channel')

plt.suptitle('RGB Channel Histogram')
plt.xlabel('Pixel Intersity')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# # 生成各通道直方图，在不同图上展示
# plt.subplot(221)
# plt.hist(r_img.ravel(),256,color='red')
# plt.title('R Channel Histogram')
# plt.xlabel('Pixel Intersity')
# plt.ylabel('Frequency')
#
# plt.subplot(222)
# plt.hist(g_img.ravel(),256,color='green')
# plt.title('G Channel Histogram')
# plt.xlabel('Pixel Intersity')
# plt.ylabel('Frequency')
#
# plt.subplot(223)
# plt.hist(b_img.ravel(),256,color='blue')
# plt.title('B Channel Histogram')
# plt.xlabel('Pixel Intersity')
# plt.ylabel('Frequency')
#
# plt.suptitle('RGB Channel Histogram')
# plt.tight_layout() #自动调整子图布局，防止子图之间或子图和图形边缘重叠
# plt.subplots_adjust(top=.85)
# plt.show()


# # 生成各通道直方图，在不同图上展示
# fig,axes = plt.subplots(2,2,figsize=(10,8))
#
# axes[0,0].hist(r_img.ravel(),256,color='red')
# axes[0,0].set_title('Red Channel Histogram')
# axes[0,0].set_xlabel('Pixel Intensity')
# axes[0,0].set_ylabel('Frequency')
#
# axes[0,1].hist(g_img.ravel(),256,color='green')
# axes[0,1].set_title('Green Channel Histogram')
# axes[0,1].set_xlabel('Pixel Intensity')
# axes[0,1].set_ylabel('Frequency')
#
# axes[1,0].hist(b_img.ravel(),256,color='blue')
# axes[1,0].set_title('Blue Channel Histogram')
# axes[1,0].set_xlabel('Pixel Intensity')
# axes[1,0].set_ylabel('Frequency')
#
# plt.suptitle('RGB Channel Histogram')
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
# plt.show()


# 方法二
# # 灰度图像直方图
# img = cv2.imread('lenna.png')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# '''
# hist = calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None) 函数：
# images：要计算直方图的图像列表，传值类型是 list（列表）
# channels：计算直方图的通道索引，传值类型是 list（列表）
# mask：掩码图像（可选），传值是 None 或 numpy.ndarray。用于选择图像中哪些部分参与直方图计算。
#      如果传递 None，则表示不使用掩码，即整个图像都会参与直方图计算。如果提供了掩码，那么只有掩码非零的部分会用于计算直方图
# histSize：每个通道的直方图大小，传值类型是 list（列表），直方图的桶数（bin），使用 [256] 表示计算包含 256 个值（即 0 到 255）的直方图
# ranges：每个通道的像素值范围（横轴范围），传值类型是 list（列表），[0, 256] 表示像素值在 0 到 255 之间
# accumulate：是否累积直方图（可选），True/False，默认值是 False 表示不累积，每次计算时从头开始
#
# 输出hist：返回的是一个 256 行 1 列的数组。hist[0] 表示像素值为 0 的像素出现的次数
# '''
# hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# plt.figure(figsize=(10,8))
# plt.title('Gray Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()


# # 彩色图像直方图
# img = cv2.imread('lenna.png')
# channels=cv2.split(img)  # cv2.split  分离图像的颜色通道，返回各通道单独的数组 (B, G, R)
# colors = ('blue', 'green', 'red')
# labels = ('Blue Channel','Green Channel','Red Channel')
#
# plt.figure(figsize=(10,8))
# plt.title('Flattend Color Histogram')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
#
# '''
# zip(*iterables) 函数用于将多个可迭代对象（如列表、元组、字符串等）的元素逐个配对
# 返回一个迭代器，其中每个元素是一个由传入可迭代对象中对应位置的元素组成的元组
#
# zip(*zipped) （zipped：已配对的列表）。将已经配对好的列表，重新拆分为多个单独的列表
# '''
# for channel,color,label in zip(channels,colors,labels):
#     hist = cv2.calcHist([channel],[0],None,[256],[0,256])
#     plt.plot(hist,color=color,label=label)
#     plt.legend()  # 显示图例
#     plt.xlim([0,256]) #设置 x 轴范围
#
# plt.show()


