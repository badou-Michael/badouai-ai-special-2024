import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
shape = img.shape[:2]
# (1) 直接调用函数将图片转换成灰度图
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# (2) 通过关系式将图片转换成灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = np.zeros((shape[0], shape[1]), dtype=np.uint8)
for i in range(shape[0]):
    for j in range(shape[1]):
        img_gray[i][j] = 0.3*img[i][j][0] + 0.59*img[i][j][1] + 0.11*img[i][j][2]

# 转换成黑白图
img_bw = np.zeros((shape[0], shape[1]), dtype=np.uint8)
for i in range(shape[0]):
    for j in range(shape[1]):
        if (img_gray[i, j]/255) <= 0.5:
            img_bw[i, j] = 0
        else:
            img_bw[i, j] = 255

# 输出原图、灰度图和黑白图
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
img_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Lenna")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_gray)
plt.title("Lenna_gray")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_bw)
plt.title("Lenna_black and white")
plt.axis("off")

plt.show()
