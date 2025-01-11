#  双线性插值法
import cv2
import numpy as np


def bilinear_interoplation(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]  # h代表Y轴 ，w代表X轴
    print("src_h, src_w =", src_h, src_w)
    print("dst_h, dst_w =", dst_h, dst_h)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    dst_img = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 根据中心点对齐找到原始图片的x0,y0的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                if src_x < 0:
                    src_x = 0
                if src_y < 0:
                    src_y = 0
                # np.floor()返回不大于输入参数的最大整数。（向下取整
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))

                # 找到插值计算的x1,y1点的坐标
                src_x1 = int(min(src_x0 + 1, src_w - 1))
                src_y1 = int(min(src_y0 + 1, src_h - 1))

                # 根据四个点坐标计算插值, 在X轴上计算得到两个点
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                # 根据Y轴上计算得到的两个点在Y轴插值一次， 得到最终的点
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interoplation(img, (900, 900))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
