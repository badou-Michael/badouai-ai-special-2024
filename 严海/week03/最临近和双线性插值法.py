import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = height / 800
    sw = height / 800
    for i in range(800):
        for j in range(800):
            h = int(i * sh + 0.5)
            w = int(j * sw + 0.5)
            emptyImage[i, j] = img[h, w]
    return emptyImage


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
                # 查找 DST 图像 X 和 Y 的原点 X 和 Y 坐标
                # 使用几何中心对称
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 查找将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


# cv2.resize(img, (800,800,c),near/bin)
if __name__ == '__main__':
    # 最临近插值法
    img = cv2.imread("lenna.png")
    zoom = function(img)
    print(zoom)
    print(zoom.shape)
    cv2.imshow("nearest interp", zoom)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # 双线插值法
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
