# 标题：week3 作业
# 作者：张瑞超
import cv2
import numpy as np

# Implementation of nearest interpolation
def nearest_interp(img, dim):
    height, width, channels = img.shape
    target_width, target_height = dim

    # 创建目标图像并计算缩放因子
    empty_img = np.zeros((target_width, target_height, channels), np.uint8)
    scale_height, scale_width = target_height / height, target_width / width

    # 向量化处理
    x_indices = np.clip((np.arange(target_height) / scale_height + 0.5).astype(int), 0, height - 1)
    y_indices = np.clip((np.arange(target_width) / scale_width + 0.5).astype(int), 0, width - 1)

    # NumPy广播来赋值，减少循环次数
    empty_img[:,:] = img[x_indices[:, None], y_indices]
    return empty_img

# Implementation of bi-linear interpolation
def bilinear_interpolation(img, dim):
    src_h, src_w, channels = img.shape
    dst_w, dst_h = dim

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 创建目标图像
    dst_img = np.zeros((dst_h, dst_w, channels), dtype=img.dtype)

    # 计算缩放因子
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    # 生成目标图像的网格坐标
    dst_x = np.arange(dst_w)
    dst_y = np.arange(dst_h)
    dst_x, dst_y = np.meshgrid(dst_x, dst_y)

    # 计算在原图像中的坐标
    src_x = (dst_x + 0.5) * scale_x - 0.5
    src_y = (dst_y + 0.5) * scale_y - 0.5

    # 找到四个邻近像素的整数坐标
    src_x0 = np.floor(src_x).astype(int)
    src_x1 = np.clip(src_x0 + 1, 0, src_w - 1)
    src_y0 = np.floor(src_y).astype(int)
    src_y1 = np.clip(src_y0 + 1, 0, src_h - 1)

    # 计算插值权重
    wa = (src_x1 - src_x) * (src_y1 - src_y)
    wb = (src_x1 - src_x) * (src_y - src_y0)
    wc = (src_x - src_x0) * (src_y1 - src_y)
    wd = (src_x - src_x0) * (src_y - src_y0)

    # 对每个通道进行插值
    for i in range(channels):
        dst_img[:, :, i] = (wa * img[src_y0, src_x0, i] +
                            wb * img[src_y1, src_x0, i] +
                            wc * img[src_y0, src_x1, i] +
                            wd * img[src_y1, src_x1, i]).astype(np.uint8)

    return dst_img

# Implementation of histogram equalization
def histogram_equalization(img):
    # 计算图像的直方图
    histogram = np.zeros(256, dtype=int)
    for pixel in img.ravel():  # 展平数组，统计每个灰度值的频率
        histogram[pixel] += 1

    # 计算累积分布函数 (CDF)
    cdf = histogram.cumsum()  # 计算累加和
    cdf_normalized = cdf * (255 / cdf[-1])  # 归一化到 [0, 255]

    # 根据 CDF 映射原始像素值到均衡化后的像素值
    equalized_img = np.interp(img.ravel(), np.arange(256), cdf_normalized).reshape(img.shape).astype(np.uint8)

    return equalized_img


img = cv2.imread('lenna.png')
dim = (800,800)
near_zoom = nearest_interp(img, dim)
bilinear_zoom = bilinear_interpolation(img, dim)
cv2.imshow('nearest interp', near_zoom)
cv2.imshow('bilinear interp', bilinear_zoom)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = histogram_equalization(gray)
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

cv2.imshow('original', img)
cv2.waitKey(0)