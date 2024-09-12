import cv2
import matplotlib.pyplot as plt
import numpy as np
import  threading


'''
绘制(灰度)直方图【法一】：1.加载图像imread, 2.计算直方图cv2.calcHist (256, 1)的数组, 3.plt绘制直方图，(256, 1)的数组用plot画，转一维数组用bar画
'''
# def get_gray_hist_mth1():
original_img = cv2.imread("lenna.png")

# 计算灰度图
gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

# 计算灰度直方图 (图像列表, 通道列表, 掩码(一般None), 直方图大小的列表(一般表示灰度级/值时每个灰度值一个bin)， 直方图的像素值范围列表(横轴))
# 直方图大小的列表(一般指灰度级/值)："bin"（也称为“bucket”或“区间”）是指将连续的数据值范围分割成一系列不重叠的子区间。每个这样的子区间就被称为一个“bin”。
gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])  # 用于计算一个或多个数组的直方图, 返回一个形状为 (256, 1) 的数组
print(gray_hist, type(gray_hist))

# 绘制直方图
# 创建独立的绘图窗口"Gray Histogram - Method 1"
plt.figure("Gray Histogram - Method 1")
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# plt.plot(gray_hist)  # plt.plot()用于绘制线图,输入256*1的数组

flat_data = gray_hist.flatten() # 将图像展平为一维数组
plt.bar(range(256), flat_data)  # 使用bar绘制柱状图
print(flat_data, type(gray_hist))

plt.xlim([0, 256])  # 设置x轴的范围为0到255


'''
绘制(灰度)直方图【法二】：1.加载图像imread, 2.准备数据(一维)， 3.plt.hist 绘制直方图
'''
# def get_gray_hist_mth2():
original_img = cv2.imread("lenna.png")

# 计算灰度图
gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

# 准备数据
data = gray_img.ravel()  # 二维数组转成一维【法一】
# data_2 = gray_img.flatten()  # 二维数组转成一维【法一】

print(gray_img, type(gray_img))
print(data, type(gray_img))
# print(data==data_2)

# 绘制直方图
# 创建独立的绘图窗口"Gray Histogram - Method 1"
plt.figure("Gray Histogram - Method 2")
plt.hist(data, 256)  # plt.hist 绘制直方图
# plt.plot(data, 256)  # ValueError: x and y must have same first dimension, but have shapes (262144,) and (1,)


'''
绘制(彩色)直方图【法一】：1.加载图像imread 2.准备数据：直方图数据列表， 3.绘制直方图
'''
# def get_color_hist_mth1(image_path):
original_img = cv2.imread("lenna.png")
orig_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

channels = (0, 1, 2)
histograms = []  # 直方图列表
colors = ('red', 'green', 'blue')

# 对于每个颜色通道
for channel in channels:
    # 计算直方图
    hist = cv2.calcHist([orig_img_rgb], [channel], None, [256], [0, 256])
    histograms.append(hist)

# 绘制直方图
plt.figure("Color Histogram - Method 1")
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for hist, color in zip(histograms, colors):
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


'''
绘制(彩色)直方图【法二】：1.加载图像imread, 2.分离图像， 3.对每个图像绘制直方图
'''
# def get_color_hist_mth2(image_path):
original_img = cv2.imread("lenna.png")

colors = ('blue', 'green', 'red')
chans = cv2.split(original_img)
# 绘制直方图
plt.figure("Color Histogram - Method 2")
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for chan, color in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()


# if __name__ == '__main__':
#
#     get_gray_hist_mth1()
#     get_gray_hist_mth2()


