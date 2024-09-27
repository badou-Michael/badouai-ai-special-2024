import cv2
import numpy as np

# 先读取原图并获取其宽高
lenna = cv2.imread("./lenna.png")
w, h = lenna.shape[ : 2]

# 将像素数据转化为数组
lennaArray = np.array(lenna)

# 创建一个800*800像素数组，存放放大后的图像
bigLennaArray = np.zeros((1200, 1200, lennaArray.shape[2]), dtype = np.uint8)

# 计算放大率(长宽相同故只计算一次)
scale = 1200 / w

# 进行最临近插值
for new_y in range(1200):
    for new_x in range(1200):
        x = int(new_x / scale + 0.5)
        y = int (new_y / scale + 0.5)
        # 边界处理，避免数据越界错误
        x = min(max(x, 0), lenna.shape[1] - 1)
        y = min(max(y, 0), lenna.shape[0] - 1) 
        bigLennaArray[new_x, new_y] = lennaArray[x, y]

# 保存放大后的图片
cv2.imwrite("./bigLenna.png", bigLennaArray)