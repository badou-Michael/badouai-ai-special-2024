import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
彩色图像直方图
"""
img = cv2.imread("lenna.png", 1)

chans = cv2.split(img)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened Color Historgram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0,256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()


"""
彩色图像直方图均衡化
"""
# 彩色图像均衡化，需要分解通道，对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并每一个通道
result = cv2.merge((bH, gH, rH))

# 使用matplotlib显示图像
plt.figure(figsize=(10, 5))   # 设置图像大小
plt.subplot(1, 2, 1)     # 1行2列的第1个位置
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    # 将BGR转换为RGB
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
