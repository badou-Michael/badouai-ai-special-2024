import numpy as np
import cv2

def bilinear(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    print(src_h, src_w)
    print(dst_h, dst_w)
    if dst_h == src_h and dst_w ==src_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), img.dtype)
    # 寻找对应比例关系
    scale_x = float(src_w / dst_w)
    scale_y = float(src_h / dst_h)
    # scale_x = src_w / dst_w
    # scale_y = src_h / dst_h
    # print(scale_x)
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心重合+通过比例关系找到img上的“虚拟点”坐标
                src_x = scale_x * (dst_x + 0.5) - 0.5
                src_y = scale_y * (dst_y + 0.5) - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 插值参照点坐标若先写x后写y，则输出的dst_img会向左旋转90°→数组取值的顺序是先y后x，先x后y就是反着取了
                # temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]
                # temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

img = cv2.imread("lenna.png")
dst_bilinear = bilinear(img, (700, 700))
cv2.imshow('bilinear interp', dst_bilinear)
# cv2.waitKey()

# 调用接口实现最邻近插值
auto_dst_bilinear = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
cv2.imshow("dst_bilinear interpolation", auto_dst_bilinear)
cv2.waitKey()
