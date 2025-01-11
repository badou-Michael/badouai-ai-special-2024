#最近邻差值
import cv2
import numpy as np

def nearest_neighbor_interpolation(img, scale):
    # 获取原始图像的尺寸
    height, width, channels = img.shape
    # 计算新图像的尺寸
    new_height, new_width = int(height * scale), int(width * scale)
    # 创建一个新图像的空白矩阵
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # 进行最近邻插值
    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的最近邻像素的索引
            x = int(i / scale)
            y = int(j / scale)
            # 将最近邻像素的值赋给新图像
            new_img[i, j] = img[x, y]

    return new_img

# 读取图像
img = cv2.imread('lenna.png')
# 检查图像是否成功读取
if img is None:
    print("Error: Image not found.")
else:
    # 放大图像，这里以放大2倍为例
    scale_factor = 2
    zoom_img = nearest_neighbor_interpolation(img, scale_factor)
    # 保存放大后的图像
    cv2.imwrite('zoomed_lenna.png', zoom_img)


##双线性差值
import cv2
import numpy as np

def bilinear_interpolation(img, scale):
    # 获取原始图像的尺寸
    height, width, channels = img.shape
    # 计算新图像的尺寸
    new_height, new_width = int(height * scale), int(width * scale)
    # 创建一个新图像的空白矩阵
    new_img = np.zeros((new_height, new_width, channels), dtype=np.float32)

    # 进行双线性插值
    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的像素位置
            x = i / scale
            y = j / scale

            # 计算四个最近邻像素的坐标
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, height - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, width - 1)

            # 计算距离权重
            x_ratio = x - x0
            y_ratio = y - y0

            # 对四个像素进行加权平均
            for c in range(channels):
                new_img[i, j, c] = (img[x0, y0, c] * (1 - x_ratio) * (1 - y_ratio) +
                                    img[x1, y0, c] * x_ratio * (1 - y_ratio) +
                                    img[x0, y1, c] * (1 - x_ratio) * y_ratio +
                                    img[x1, y1, c] * x_ratio * y_ratio)

    return new_img.astype(np.uint8)

# 读取图像
img = cv2.imread('lenna.png')
# 检查图像是否成功读取
if img is None:
    print("Error: Image not found.")
else:
    # 放大图像，这里以放大2倍为例
    scale_factor = 2
    zoom_img = bilinear_interpolation(img, scale_factor)
    # 保存放大后的图像
    cv2.imwrite('zoomed_lenna_bilinear.png', zoom_img)