import numpy as np
import cv2


def bilinear(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    # 判断图像是否发生改变，若没改变则直接复制
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 创建新图像
    dst_img = np.zeros((dst_h, dst_w, channels), img.dtype)
    # 计算图形缩放比例
    scale_x = float(dst_w/src_w)
    scale_y = float(dst_h/src_h)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算新图像在原图上的"虚拟"对应点
                src_x = (dst_x + 0.5)/scale_x - 0.5
                src_y = (dst_y + 0.5)/scale_y - 0.5

                # 计算图像点不能超出原图像范围
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 图像旋转
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    dst_bilinear = bilinear(img, (700, 700))
    cv2.imshow("src_img", img)
    cv2.imshow("dst_bilinear", dst_bilinear)
    #cv2.waitKey(0)

    # cv2 接口调用实现差值
    auto_bilinear = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("auto_bilinear", auto_bilinear)
    cv2.waitKey(0)
