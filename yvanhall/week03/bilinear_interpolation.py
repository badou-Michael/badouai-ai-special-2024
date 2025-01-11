# -*- coding: utf-8 -*-

import numpy as np
import cv2

# 双线性插值算法
def bilinear_interpolation(img, out_img):
    src_h, src_w , channels = img.shape
    res_h, res_w = out_img[1], out_img[0]

    if src_h == res_h and src_w == res_w:
        return img.copy()
    dst_img = np.zeros((res_h, res_w, 3), dtype=np.uint8)

    # 倍数
    scale_w = src_w / res_w
    scale_h = src_h / res_h

    for i in range(channels):
        for res_y in range(res_h):
            for res_x in range(res_w):
                # 几何中心对齐
                src_y = (res_y + 0.5) * scale_h - 0.5
                src_x = (res_x + 0.5) * scale_w - 0.5
                # 边界判定
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 双线性插值公式
                line1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                line2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[res_y, res_x, i] = int((src_y1 - src_y) * line1 + (src_y - src_y0) * line2)

    return dst_img



if __name__ == '__main__':
    origin_img = cv2.imread("lenna.png")
    # cv2双线性插值算法
    # result_img = cv2.resize(origin_img, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    result_img = bilinear_interpolation(origin_img, (1000, 1000))
    cv2.imshow("result image", result_img)
    cv2.waitKey(0)