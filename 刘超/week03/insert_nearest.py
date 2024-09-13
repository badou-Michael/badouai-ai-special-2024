import numpy as np
from PIL import Image

def insert_nearst(img, new_width, new_height):
    # 获取原始图像的宽度和高度
    height, width = img.shape[:2]

    # 创建一个新的图像数组
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 计算缩放比例
    x_scale = width / new_width
    y_scale = height / new_height

    for i in range(new_width):
        for j in range(new_height):
            # 计算原始图像中对应的坐标
            x = int(i * x_scale)
            y = int(j * y_scale)
            new_img[i, j] = img[x, y]
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
resized_image_array = insert_nearst(image_array, new_width, new_height)

# 将 NumPy 数组转换回图像
resized_image = Image.fromarray(resized_image_array)

# 显示原始图像和放大后的图像
image.show()
resized_image.show()

# 保存放大后的图像
resized_image.save('insert_nerast_resized.jpg')