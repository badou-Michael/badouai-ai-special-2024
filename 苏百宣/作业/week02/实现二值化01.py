# 实现二值化01
# author：苏百宣
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io

# 读取图像并转换为灰度图像
img = io.imread('lenna.png')
img_gray = rgb2gray(img)
# 获取图像的行数和列数
row, cols = img_gray.shape
# 对图像进行二值化处理
for i in range(row):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
# 显示二值化后的图像
plt.imshow(img_gray, cmap='gray')
plt.show()

# 打印二值化后的图像矩阵
print("---Binary Image Matrix---")
print(img_gray)

