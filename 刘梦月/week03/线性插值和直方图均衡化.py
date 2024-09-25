# 最临近插值法
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_neighbor(img, scale=1):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x, y = int(i / scale), int(j / scale)
            new_img[i, j] = img[x, y]
    return new_img

img = cv2.imread('cat.jpg')
new_img = nearest_neighbor(img, 0.5)
cv2.imshow('original', img)
cv2.imshow('new', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('new_cat_linear.jpg', new_img)


# 双线性插值法
def bilinear_interpolation(img, scale=1):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale
            x1, y1 = int(x), int(y)
            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)
            # 计算插值权重
            a = x - x1
            b = y - y1
            # 计算插值结果
            for c in range(img.shape[2]):
                # 处理每个通道
                I11 = img[x1, y1, c]
                I12 = img[x1, y2, c]
                I21 = img[x2, y1, c]
                I22 = img[x2, y2, c]
                new_img[i, j, c] = (1 - a) * (1 - b) * I11 + (1 - a) * b * I12 + a * (1 - b) * I21 + a * b * I22
    return new_img
img = cv2.imread('cat.jpg')
new_img = bilinear_interpolation(img, 0.5)
cv2.imshow('original', img)
cv2.imshow('new', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('new_cat_bilinear.jpg', new_img)

# 直方图均衡化
img = cv2.imread('cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray)

# 显示原始图像和均衡化后的图像
cv2.imshow('original', img)
cv2.imshow('equ', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('new_cat_hist.jpg', equ)

# 显示均衡化前后的直方图
plt.subplot(121), plt.hist(gray.ravel(), 256, [0, 256]), plt.title('Before')
plt.subplot(122), plt.hist(equ.ravel(), 256, [0, 256]), plt.title('After')
plt.savefig('hist.png')
plt.show()
