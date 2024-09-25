"""
最邻近插值法
"""
import cv2
import numpy as np


def nearest(image, out_height, out_width):
    height, width, channels = image.shape
    empty_image = np.zeros((out_height, out_width, channels), np.uint8)
    sh = out_height / height  # 算出放大比例
    sw = out_width / width
    for i in range(out_height):
        for j in range(out_width):
            empty_image[i, j] = image[int(i / sh + 0.5), int(j / sw + 0.5)]  # 通过除以放大比例来确定i和j在原图位置上的像素值
    return empty_image


lenna_image = cv2.imread("../week02/lenna.png")
output_image = nearest(lenna_image, 800, 800)
print(output_image)
print(output_image.shape)
cv2.imshow("output_image", output_image)
cv2.imshow("lenna_image", lenna_image)
cv2.waitKey(0)
