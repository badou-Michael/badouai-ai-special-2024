import numpy as np
import cv2


def function(image, out_dimension):
    src_height, src_width, channel = image.shape
    print(channel)
    dst_height, dst_width = out_dimension[0], out_dimension[1]
    if src_height == dst_height and src_width == dst_width:
        return image.copy()
    dst_image = np.zeros((dst_height, dst_width, channel), dtype=np.uint8)
    print(dst_image.shape)
    scale_height, scale_width = float(src_height) / dst_height, float(src_width) / dst_width
    for i in range(channel):
        for dst_x in range(dst_height):
            for dst_y in range(dst_width):
                src_x = (dst_x + 0.5) * scale_height - 0.5
                src_y = (dst_y + 0.5) * scale_width - 0.5
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, src_height - 1)
                src_y1 = min(src_y0 + 1, src_width - 1)
                temp0 = (src_y1 - src_y) * image[src_x0, src_y0, i] + (src_y - src_y0) * image[src_x0, src_y1, i]
                temp1 = (src_y1 - src_y) * image[src_x1, src_y0, i] + (src_y - src_y0) * image[src_x1, src_y1, i]
                dst_image[dst_x, dst_y, i] = int((src_x1 - src_x) * temp0 + (src_x - src_x0) * temp1)

    return dst_image


image = cv2.imread('moon.JPG')
dst_image = function(image, (800, 800))
cv2.imshow("binary interpolation", dst_image)
cv2.waitKey(0)
