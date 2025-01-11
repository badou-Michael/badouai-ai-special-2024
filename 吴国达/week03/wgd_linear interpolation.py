import cv2
import numpy as np

'''
双线性插值
'''


def linear_interpolation(img, out_dim):
    s_h, s_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    # 判断如果宽高与原图一样则返回原图
    if s_h == dst_h and s_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(s_w) / dst_w, float(s_h) / dst_h

    for i in range(channel):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                # 中心重合
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, s_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, s_h - 1)

                # 计算插值
                fr1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                fr2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * fr1 + (src_y - src_y0) * fr2)
        print(dst_img)
    return dst_img


src_image = cv2.imread('images/lenna.png')
dst_image = linear_interpolation(src_image, (900, 900))
# dst = cv2.resize(src_image, (900, 900), cv2.INTER_LINEAR)
cv2.imshow('linear interp', dst_image)
cv2.waitKey(0)
