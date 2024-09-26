
### 作业1.实现最邻近插值和双线性插值

## 1.1 最邻近插值
import cv2
import numpy as np

def function(img):
    height, width, channels = img.shape
    empty_image = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width

    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            empty_image[i, j] = img[x, y]

    return empty_image


img = cv2.imread("lenna.png")
zoom = function(img)
cv2.imshow(winname="original image", mat=img)
cv2.imshow(winname="after zoom", mat=zoom)
cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows()

## 1.2 双线性插值
import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 几何中心重合的处理
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找出将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 代入公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear innterp', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### 作业3.直方图均衡化。
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

