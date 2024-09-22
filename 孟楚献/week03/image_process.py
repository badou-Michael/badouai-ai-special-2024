import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.summary import image
from tensorflow import int16, newaxis, meshgrid


#双线性插值
def optimized_bilinear_interpolation(img, out_dim):
    start_time = datetime.datetime.now()
    print(f"开始执行函数 {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    src_h, src_w, channels = img.shape
    des_h, des_w = out_dim

    # 创建目标坐标网格
    x = np.linspace(0, src_w - 1, des_w)
    y = np.linspace(0, src_h - 1, des_h)
    x, y = np.meshgrid(x, y)

    # 计算相邻像素的坐标
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)

    # 计算插值权重
    wx = x - x0
    wy = y - y0

    # 执行双线性插值
    t0 = (1 - wy)[:, :, np.newaxis] * (img[y0, x0] * (1 - wx)[:, :, np.newaxis] + img[y0, x1] * wx[:, :, np.newaxis])
    t1 = wy[:, :, np.newaxis] * (img[y1, x0] * (1 - wx)[:, :, np.newaxis] + img[y1, x1] * wx[:, :, np.newaxis])

    des_img = t0 + t1
    end_time = datetime.datetime.now()
    print(f"结束执行函数 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return des_img.astype(img.dtype)

# 最邻近插值
def nearest_interpolation(image, shape):
    height, width, channel = image.shape
    sh = float(shape[0]) / height
    sw = float(shape[1]) / width

    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    # new_image = np.zeros((shape[0], shape[1], channel), dtype=image.dtype)
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         x = min(int(i / sh + 0.5), height - 1)
    #         y = min(int(j / sw + 0.5), width - 1)
    #         new_image[i][j] = image[x][y]
    # 计算对应的源图像坐标
    x = np.minimum(np.array(i / sh + 0.5, dtype=int), height - 1)
    y = np.minimum(np.array(j / sw + 0.5, dtype=int), width - 1)

    # 使用向量化索引获取新图像的值
    new_image = image[x, y]

    return new_image

#直方图均衡化 预期参数为灰度图
def histogram_equalization(image:np.ndarray):
    vals, counts = np.unique(image, return_counts=True)
    # 像素点个数
    pix_count = image.shape[0] * image.shape[1]
    # 当前计算过的像素点总数
    cur_total_count = 0
    #存储原像素值到新像素值的映射
    mapping = {}
    for value, count in zip(vals, counts):
        cur_total_count += count
        q = (cur_total_count / pix_count * 256 - 1)
        q = q if q > 0 else 0
        mapping[value] = q
        print(f"值 {value} 出现了 {count} 次，均衡化后的值为 {q}")
    return np.vectorize(lambda x : mapping[x])(image)

image = cv2.imread("img1.png")
cv2.imshow('原图', image)

nearest_image = nearest_interpolation(image, (200, 209))
bilinear_image = optimized_bilinear_interpolation(image, (200, 200))

grey_image = np.dot(image, ([0.11, 0.59, 0.30]))
#调暗
grey_image *= 0.5
grey_image[0] = 255

new_grey_image = histogram_equalization(grey_image)
print(new_grey_image)

# 设置字体为 SimHei
plt.rcParams['font.family'] = 'SimHei'
# 创建一个 2x2 的子图
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# 将所有子图的坐标轴关闭
for ax in axs.flat:
    ax.axis('off')

# 在每个子图上显示图片
axs[0, 0].imshow(cv2.cvtColor(nearest_image, cv2.COLOR_BGR2RGB), cmap='gray')
axs[0, 0].set_title('最邻近插值')

axs[0, 1].imshow(cv2.cvtColor(bilinear_image, cv2.COLOR_BGR2RGB), cmap='gray')
axs[0, 1].set_title('双线性插值')

axs[1, 0].imshow(grey_image, cmap='gray')
axs[1, 0].set_title('灰度图')

axs[1, 1].imshow(new_grey_image, cmap='gray')
axs[1, 1].set_title('均衡化后灰度图')

plt.show()
cv2.waitKey(0)



