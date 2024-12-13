# 实现第一种hash算法
# author:苏百宣
import cv2
import numpy as np


# 实现均值哈希算法
def aHash(img):
    # 调整大小到 9x9
    img = cv2.resize(img, (9, 9), interpolation=cv2.INTER_CUBIC)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 调试：打印灰度图
    print(f"Gray Image:\n{gray}")

    # 初始化累加值和哈希字符串
    s = 0.0  # 使用浮点数避免溢出
    hash_str = ''

    # 计算灰度值总和
    for i in range(9):
        for j in range(9):
            s += gray[i][j]

    # 计算灰度均值
    avg = s / 81

    # 调试：打印灰度均值
    print(f"Average Gray Value: {avg}")

    # 生成哈希字符串
    for i in range(9):
        for j in range(9):
            if gray[i][j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


# 加载图片
img = cv2.imread('/Users/ferry/Desktop/八斗作业/week08/sww1028.jpg')

# 检查图片是否加载成功
if img is None:
    raise ValueError("Image not found or unable to load!")

# 计算并打印哈希值
hash_value = aHash(img)
print(f"Hash Value: {hash_value}")

# 显示图片
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



