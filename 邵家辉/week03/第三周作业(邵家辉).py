import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1、实现最临近插值和双线性插值
# 1.1 最临近插值
img = cv2.imread("lenna.png")
height, width, chanel = img.shape
emptyimage = np.zeros((800,800,chanel),np.uint8)
hw = 800/height
ww = 800/width
for i in range(800):
    for j in range(800):
        a = int(i/hw + 0.5)
        b = int(j/ww + 0.5)
        emptyimage[i,j] = img[a,b]
cv2.imshow("this si ",emptyimage)
cv2.waitKey(0)

# 1.2 双线性插值
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
# if __name__ == '__main__':
img = cv2.imread('lenna.png')
dst = bilinear_interpolation(img, (700, 700))
cv2.imshow('bilinear interp', dst)
cv2.waitKey()




# 2、证明中心重合+0.5
# 证明的图片链接如下：
# https://easylink.cc/1b0isc




# 3、实现直方图均衡化
image = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
cv2.imshow("this is ", np.hstack([gray, dst]))
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
