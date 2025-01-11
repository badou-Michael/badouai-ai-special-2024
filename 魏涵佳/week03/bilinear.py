import cv2
import numpy as np


def bilinear_interpolation(img, scale_y, scale_x):
    if scale_y == 1 and scale_x == 1:
        return img

    h, w, c = img.shape
    dist_h, dist_w = int(h * scale_y), int(w * scale_x)
    dist_img = np.zeros((dist_h, dist_w, c), dtype=np.uint8)

    # 此处必须要对通道数及逆行循环，因为在使用int整数转换时，只能对numpy标量进行操作
    for i in range(c):
        for dist_y in range(dist_h):
            for dist_x in range(dist_w):
                src_x = (dist_x + 0.5) / scale_x - 0.5
                src_y = (dist_y + 0.5) / scale_y - 0.5

                # find the nearest 4 pixels in the original image
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, w - 1)

                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, h - 1)

                # bilinear interpolation
                value0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                value1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                dist_img[dist_y, dist_x, i] = int((src_y1 - src_y) * value0 + (src_y - src_y0) * value1)
    return dist_img

'''
双层循环的方式：
    for dist_y in range(dist_h):
        for dist_x in range(dist_w):
            src_x = (dist_x + 0.5) / scale_x - 0.5
            src_y = (dist_y + 0.5) / scale_y - 0.5

            # find the nearest 4 pixels in the original image
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, w - 1)

            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, h - 1)

            # calculate interpolation weights
            fx1 = src_x - src_x0
            fx0 = 1 - fx1
            fy1 = src_y - src_y0
            fy0 = 1 - fy1

            # bilinear interpolation
            value0 = fx0 * img[src_y0, src_x0] + fx1 * img[src_y0, src_x1]
            value1 = fx0 * img[src_y1, src_x0] + fx1 * img[src_y1, src_x1]

            dist_img[dist_y, dist_x] = fy0 * value0 + fy1 * value1

'''

if __name__ == "__main__":
    img = cv2.imread('../imgs/Lenna.png')
    resize_img = bilinear_interpolation(img, 1.5, 1.5)

    cv2.imshow('Original Image', img)
    cv2.imshow('Bilinear Image', resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
