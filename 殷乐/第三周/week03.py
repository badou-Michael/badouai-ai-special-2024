import cv2
import numpy as np
from matplotlib import pyplot as plt


# 最邻近插值：输入图像和目标宽高，返回新图像
def nearest_neighbor_interpolation(src_img, out_dim):
    # 获取到目标图像的宽高
    dst_width, dst_height = out_dim[0], out_dim[1]
    # 获取原图尺寸信息
    src_height, src_width, channels = src_img.shape
    # 创建新图全0矩阵
    dst_img = np.zeros((dst_height, dst_width, channels), dtype=np.uint8)
    # 获取放大/缩小系数
    scale_width = float(src_width) / dst_width
    scale_height = float(src_height) / dst_height
    # 遍历赋值
    for dsth in range(dst_height):
        for dstw in range(dst_width):
            # 求目标点在原图上的位置并向下取整
            src_x = int(dsth * scale_height + 0.5)
            src_y = int(dstw * scale_width + 0.5)
            dst_img[dsth, dstw] = src_img[src_x, src_y]
    # print(dst_img.shape)
    return dst_img


# 双线性插值：输入图像和目标宽高，返回新图像
def bilinear_interpolation(src_img, out_dim):
    # 获取目标图像宽高
    dst_width, dst_height = out_dim[0], out_dim[1]
    # 获取原图尺寸
    src_height, src_width, channels = src_img.shape
    # 如果图像大小一致，直接返回
    if src_height == dst_height and src_width == dst_width:
        return src_img.copy()
    # 创建新全零矩阵
    dst_img = np.zeros((dst_height, dst_width, channels), dtype=np.uint8)
    # 获取缩放比例
    scale_width = float(src_width) / dst_width
    scale_height = float(src_height) / dst_height
    # 遍历赋值
    for channel in range(channels):
        for dsth in range(dst_height):
            for dstw in range(dst_width):
                # 中心对齐
                src_x = (dsth + 0.5) * scale_height - 0.5
                src_y = (dstw + 0.5) * scale_width - 0.5
                # 获取临近点坐标
                x0 = int(np.floor(src_x))  # 小坐标向下取整：np.floor()返回不大于输入参数的最大整数
                x1 = min(x0 + 1, src_width - 1)  # 大坐标避免超出图像
                y0 = int(np.floor(src_y))
                y1 = min(y0 + 1, src_width - 1)
                # print(x0,x1,y0,y1)
                # 计算像素插值 y=(x1-x)*y0+(x-x0)*y1
                # (x0,y0)  tmp1  (x0,y1)
                #          dst
                # (x1,y0)  tmp2  (x1,y1)
                tmp1 = (y1 - src_y) * src_img[x0, y0, channel] + (src_y - y0) * src_img[x0, y1, channel]
                tmp2 = (y1 - src_y) * src_img[x1, y0, channel] + (src_y - y0) * src_img[x1, y1, channel]
                dst_img[dsth, dstw, channel] = int((x1 - src_x) * tmp1 + (src_x - x0) * tmp2)
    return dst_img


# 直方图均衡化
def histogram_equalizeHist(img):
    # 灰度处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 均衡化
    dst = cv2.equalizeHist(img_gray)

    plt.subplot(221)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(222)
    plt.hist(img_gray.ravel(), 256)
    plt.subplot(223)
    plt.imshow(dst, cmap='gray')
    plt.subplot(224)
    plt.hist(dst.ravel(), 256)
    plt.show()


# 彩色图像均衡化
def color_histogram_equalizeHist(img):
    # 通道分解
    (channel_b, channel_g, channel_r) = cv2.split(img)
    # 分别均衡化
    bH = cv2.equalizeHist(channel_b)
    gH = cv2.equalizeHist(channel_g)
    rH = cv2.equalizeHist(channel_r)
    # 合并通道
    dst_img = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_img", dst_img)


def func1():
    # 读图
    img_org = cv2.imread('lenna.png')
    cv2.imshow("original image", img_org)
    img_nearset_m = nearest_neighbor_interpolation(img_org, (500, 800))
    img_nearset = cv2.resize(img_org, (500, 800), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("my nearset neighbor interpolation image", img_nearset_m)
    cv2.imshow("nearset neighbor interpolation image", img_nearset)
    cv2.waitKey(0)


def func2():
    # 读图
    img_org = cv2.imread('lenna.png')
    cv2.imshow("original image", img_org)
    img_bilinear_m = bilinear_interpolation(img_org, (500, 800))
    img_bilinear = cv2.resize(img_org, (500, 800), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("my bilinear interpolation image", img_bilinear_m)
    cv2.imshow("bilinear interpolation image", img_bilinear)
    cv2.waitKey(0)


def func3():
    # 读图
    img_org = cv2.imread('lenna.png')
    cv2.imshow("original image", img_org)
    # 彩色图均衡化
    color_histogram_equalizeHist(img_org)
    # 直方图均衡化
    histogram_equalizeHist(img_org)


# 1.1 最邻近插值
# func1()

# 1.2 双线性插值
# func2()

# 3 直方图均衡化
# func3()


# 2 证明中心重合+0.5
# 设原图尺寸（M，N），缩放至（m，n）
# scale-h=M/m
# scale-w=N/n
# 几何中心分别为
# Q(X((M-1)/2) Y((N-1)/2))
# q(x((m-1)/2) y((n-1)/2))
# 如果几何中心重合，即经缩放后q=Q，所以需要添加偏移量A
# (M-1)/2+A=M/m((m-1)/2+A)
# m[(M-1)/2+A]=M[(m-1)/2+A]
# m(M-1)/2+mA=M(m-1)/2+MA
# -m/2+mA=-M/2+MA
# (M-m)/2=(M-m)A
# 所以A=0.5
# y方向同理
