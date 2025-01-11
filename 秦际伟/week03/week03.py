import cv2
import numpy as np


# 最临近插值方法
def nerest_interpolation(image, new_height, new_width):
    height, width, channels = image.shape
    empty_image = np.zeros((new_height, new_width, channels), np.uint8)
    sh = new_height/height
    sw = new_width/width
    for i in range(new_height):
        for j in range(new_width):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            # print(x, y)
            # print(image[x, y])
            empty_image[i, j] = image[x, y]
    return empty_image


# 双线性插值方法
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

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

# 直方图均衡化方法
def histogram_equalization(img):
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result

if __name__ == "__main__":
    new_height = 800
    new_width = 800
    img = cv2.imread("lenna.png")
    # 原图展示
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # 最临近插值展示
    zoom = nerest_interpolation(img, new_height, new_width)
    # print(zoom.shape)
    cv2.imshow("nearest interp", zoom)
    cv2.waitKey(0)

    # 双线性插值展示
    dst = bilinear_interpolation(img,(300,300))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey(0)

    # 直方图均衡化展示
    res = histogram_equalization(img)
    cv2.imshow("histogram equalization", res)
    cv2.waitKey(0)
