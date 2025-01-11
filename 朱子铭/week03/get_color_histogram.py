import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义获取彩色图像直方图的函数
def get_color_histogram(img):
    # 使用 OpenCV 的 split 函数将彩色图像拆分为三个通道（蓝、绿、红）
    channels = cv2.split(img)
    # 定义颜色名称的元组，分别对应蓝、绿、红三个通道
    colors = ('b', 'g', 'r')
    # 使用 matplotlib 创建一个新的图形窗口
    plt.figure()
    # 设置图形窗口的标题为"Color Histogram"（彩色图像直方图）
    plt.title("Color Histogram")
    # 设置 X 轴标签为"Bins"（区间）
    plt.xlabel("Bins")
    # 设置 Y 轴标签为"# of Pixels"（像素数量）
    plt.ylabel("# of Pixels")
    # 遍历每个通道及其对应的颜色
    for (channel, color) in zip(channels, colors):
        # 使用 OpenCV 的 calcHist 函数计算当前通道的直方图
        # 参数 [channel] 表示要计算直方图的通道，[0] 表示通道索引，None 表示不使用掩码，[256] 表示直方图的区间数，[0, 256] 表示像素值的范围
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        # 使用 matplotlib 的 plot 函数绘制当前通道的直方图，指定颜色为当前通道对应的颜色
        plt.plot(hist, color=color)
        # 设置 X 轴的范围为 0 到 256
        plt.xlim([0, 256])
    # 添加图例，显示颜色与通道的对应关系
    plt.legend(colors)
    # 显示图形窗口，展示彩色图像的直方图
    plt.show()

if __name__ == '__main__':
    # 使用 OpenCV 的 imread 函数读取名为'lenna.png'的彩色图像
    img = cv2.imread('lenna.png')
    # 使用 OpenCV 的 imshow 函数显示彩色图像，并将窗口命名为'lenna_color'
    cv2.imshow('lenna_color', img)
    # 调用 get_color_histogram 函数计算并显示彩色图像的直方图
    get_color_histogram(img)
    # 使用 OpenCV 的 waitKey 函数等待用户按键，参数 0 表示无限等待
    cv2.waitKey(0)

# # 读取名为"lenna.png"的彩色图像，第二个参数1表示以彩色模式读取
# img = cv2.imread("lenna.png", 1)
# # 将彩色图像转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 灰度图像的直方图，方法一
# plt.figure()
# # 使用matplotlib的hist函数绘制灰度图像的直方图，gray.ravel()将灰度图像展平为一维数组，256表示将像素值范围分为256个区间
# plt.hist(gray.ravel(), 256)
#
# # 灰度图像的直方图, 方法二
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure()  # 新建一个图像窗口
# plt.title("Grayscale Histogram")  # 设置图像窗口的标题为"Grayscale Histogram"
# plt.xlabel("Bins")  # 设置X轴标签为"Bins"
# plt.ylabel("# of Pixels")  # 设置Y轴标签为"# of Pixels"
# plt.plot(hist)  # 绘制直方图，以像素值区间为横坐标，对应区间的像素数量为纵坐标
# plt.xlim([0, 256])  # 设置X轴的范围为0到256
# plt.show()  # 显示图像窗口，展示绘制的直方图
