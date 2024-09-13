import numpy as np
from PIL import Image

def bilinear_interpolate(img, new_width, new_height):
    # 获取原始图像的宽度和高度
    height, width = img.shape[:2]

    # 创建一个新的图像数组
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 计算缩放比例
    x_scale = width / new_width
    y_scale = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的坐标
            x = j * x_scale
            y = i * y_scale

            # 找到四个最近邻的像素
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, height - 1)

            # 计算权重
            x_weight = x - x0
            y_weight = y - y0

            # 双线性插值
            new_img[i, j] = (img[y0, x0] * (1 - x_weight) * (1 - y_weight) +
                             img[y0, x1] * x_weight * (1 - y_weight) +
                             img[y1, x0] * (1 - x_weight) * y_weight +
                             img[y1, x1] * x_weight * y_weight)

    return new_img

# 读取图像
image_path = 'lenna.png'
image = Image.open(image_path)

# 将图像转换为 NumPy 数组
image_array = np.array(image)

# 放大图像
scale_factor = 2  # 放大两倍
new_width = image_array.shape[1] * scale_factor
new_height = image_array.shape[0] * scale_factor
resized_image_array = bilinear_interpolate(image_array, new_width, new_height)

# 将 NumPy 数组转换回图像
resized_image = Image.fromarray(resized_image_array)

# 显示原始图像和放大后的图像
image.show()
resized_image.show()

# 保存放大后的图像
resized_image.save('bilinear_resized.jpg')