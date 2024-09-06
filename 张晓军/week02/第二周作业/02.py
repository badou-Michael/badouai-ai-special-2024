import cv2
import numpy as np

# 读取彩色图像
input_image = cv2.imread('lenna.png')

# 确保图像成功加载
if input_image is None:
    print("无法加载图像文件")
else:
    # 获取图像的高度和宽度
    height, width = input_image.shape[:2]

    # 创建一个空的灰度图像
    gray_image = np.zeros((height, width), dtype=np.uint8)

    # 遍历每个像素并计算平均值
    for y in range(height):
        for x in range(width):
            # 获取像素的BGR通道值
            b, g, r = input_image[y, x]

            # 计算平均值并将其赋给灰度图像
            gray_value = 0.114*b + 0.587*g + 0.299*r
            gray_image[y, x] = gray_value


    # 显示原始图像和灰度图像
    cv2.imshow('原始图像', input_image)
    cv2.imshow('灰度图像', gray_image)



# 读取灰度图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 设置阈值
threshold_value = 128

# 应用二值化
retval, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# 显示原始图像和二值化图像

cv2.imshow('二值化图像', binary_image)

# 等待按键按下后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
