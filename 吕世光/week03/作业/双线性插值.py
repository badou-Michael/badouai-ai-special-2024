import cv2
import numpy as np


def function(img, height, width):
    src_h, src_w, src_c = img.shape
    new_img = np.zeros((height, width, src_c), np.uint8)
    scale_h = float(src_h / height)
    scale_w = float(src_w / width)
    for i in range(src_c):
        for j in range(height):
            for k in range(width):
                # 中心点对齐
                src_x = (k + 0.5) * scale_w - 0.5
                src_y = (j + 0.5) * scale_h - 0.5

                # 判断临界
                src_x_0 = int(float(src_x))
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_0 = int(float(src_y))
                src_y_1 = min(src_y_0 + 1, src_h - 1)

                # 套入公式
                parm1 = (src_x_1 - src_x) * img[src_y_0, src_x_0, i] + (src_x - src_x_0) * img[src_y_0, src_x_1, i]
                parm2 = (src_x_1 - src_x) * img[src_y_1, src_x_0, i] + (src_x - src_x_0) * img[src_y_1, src_x_1, i]
                new_img[j, k, i] = int((src_y_1 - src_y) * parm1 +
                                       (src_y - src_y_0) * parm2)

    return new_img


o_img = cv2.imread("lenna.png")
n_img = function(o_img, 800, 800)

cv2.imshow("origin", o_img)
cv2.imshow("scaleImg", n_img)
cv2.waitKey(0)
