import cv2
import numpy as np

# 读取图像
img = cv2.imread("lenna.png")

# 检查是否成功读取图像
if img is None:
    print("Error: Image not found.")
else:
    # 获取图像的高和宽
    h, w = img.shape[:2]

    # 缩小图像（缩小为原来的一半）
    resized_nearest = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    resized_linear = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    resized_cubic = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    resized_lanczos = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)

    # 显示不同插值方法的结果
    cv2.imshow("Original Image", img)
    cv2.imshow("Resized Nearest", resized_nearest)
    cv2.imshow("Resized Linear", resized_linear)
    cv2.imshow("Resized Cubic", resized_cubic)
    cv2.imshow("Resized Lanczos", resized_lanczos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()