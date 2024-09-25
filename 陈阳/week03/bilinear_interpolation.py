"""
双线性插值法

                      y - y0     x - x0                        x1 - x         x - x0
插值法推导公式：        ——————  =  ——————       ————>      y =   ——————  y0  +  ——————  y1
                      y1 - y0    x1 - x0                       x1 - x0        x1 - x0


几何中心推导过程
原图像M*M -> 目标图像N*N
中心点[(M-1)/2,(M-1)/2],[(N-1)/2,(N-1)/2]
M/N是放大缩小的比例，所以Xn = (M/N)*n
使几何中心相同，得出(M-1)/2 = (N-1)/2 * (M/N)
使两边x同时加上z，(M-1)/2 + 0.5 = ((N-1)/2 + 0.5) * (M/N)，得出z=0.5,所以两边同时加上0.5可以使几何中心对称
"""

import cv2
import numpy as np


def bilinear_interpolation(image, out_size):
    height, width, channels = image.shape
    out_height, out_width = out_size[1], out_size[0]
    if height == out_height and width == out_width:
        return image.copy()
    out_image = np.zeros((out_height, out_width, 3), np.uint8)
    scale_x, scale_y = float(height) / out_height, float(width) / out_width
    for i in range(channels):
        for out_y in range(out_height):
            for out_x in range(out_width):
                # 中心对称操作
                image_x = (out_x + 0.5) * scale_x - 0.5
                image_y = (out_y + 0.5) * scale_y - 0.5
                # 取到对应的四个点
                image_x0 = int(image_x)
                image_x1 = min(image_x0 + 1, width - 1)
                image_y0 = int(image_y)
                image_y1 = min(image_y0 + 1, height - 1)
                # 根据四个点进行三次单线性计算，得出像素值
                temp0 = ((image_x1 - image_x) * image[image_y0, image_x0, i] +
                         (image_x - image_x0) * image[image_y0, image_x1, i])
                temp1 = ((image_x1 - image_x) * image[image_y1, image_x0, i] +
                         (image_x - image_x0) * image[image_y1, image_x1, i])
                out_image[out_y, out_x, i] = int((image_y1 - image_y) * temp0 + (image_y - image_y0) * temp1)
    return out_image


if __name__ == "__main__":
    lenna_image = cv2.imread("../week02/lenna.png")
    output_image = bilinear_interpolation(lenna_image, (700, 700))
    cv2.imshow("bilinear_interpolation", output_image)
    cv2.imshow("lenna", lenna_image)
    cv2.waitKey(0)
