import cv2
import numpy as np


def bilinear_interpolation(image, out_dim):
    # 获取原图的高，宽，通道
    src_h, src_w, channel = img.shape
    # 获取目标图的高，宽
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 如果高和宽都相同的话，直接复制一份
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 创建一个空白的图像，作为目标图像
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 获取缩放的比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) /dst_w
    # 遍历目标图像的所有通道的像素
    for i in range(channel):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                # 使两个图像的几何中心重合，计算目标像素在原始图像中的坐标
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5

                # 找到 src_x 所在列的最接近的左侧像素的 x 坐标。
                src_x0 = int(np.floor(src_x))  # 向下取整
                # 确保坐标不超出图像边界
                src_x1 = min(src_x0 + 1, src_w -1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h -1)

                # 计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interpolation', dst)
    cv2.waitKey(0)














