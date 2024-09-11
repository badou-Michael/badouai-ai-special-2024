
import cv2
import numpy as np

def nearest(img, out_dim):

    height, width, channels = img.shape
    dst_width, dst_height = out_dim[0], out_dim[1]
    emptyImage = np.zeros((dst_height, dst_width, channels), np.uint8)
    sh = dst_height / height
    print(sh)
    sw = dst_width / width
    for i in range(dst_height):
        for j in range(dst_width):
            # x = int(min(i * height / dst_height + 0.5, height - 1))
            x = int(min(i / sh + 0.5, height - 1))
            y = int(min(j / sw + 0.5, width - 1))
            emptyImage[i, j] = img[x, y]
    return emptyImage


def bilinear(img, out_dim):

    height, width, channels = img.shape
    dst_height, dst_width = out_dim[0], out_dim[1]
    emptyImage = np.zeros((dst_height, dst_width, channels), np.uint8)
    sh = float(dst_height)/height
    sw = float(dst_width)/width
    for channel in range(channels):
        for i in range(dst_height):
            for j in range(dst_width):
                src_x = (i + 0.5)/sh - 0.5
                src_y = (j + 0.5)/sh - 0.5

                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, height - 1)
                src_y1 = min(src_y0 + 1, width - 1)

                dst_1 = (src_x1 - src_x) * img[src_x0, src_y0, channel] + \
                        (src_x - src_x0) * img[src_x1, src_y0, channel]
                dst_2 = (src_x1 - src_x) * img[src_x0, src_y1, channel] + \
                        (src_x - src_x0) * img[src_x1, src_y1, channel]

                emptyImage[i, j, channel] = (src_y1 - src_y) * dst_1 +\
                                            (src_y - src_y0) * dst_2
    return emptyImage

img = cv2.imread('lenna.png',1)
near_result = nearest(img,(800,800))
print(near_result.shape)
cv2.imshow("nearest interp", near_result)
# cv2.waitKey(0)

bilinear_result = bilinear(img, (800, 800))
print(bilinear_result.shape)
cv2.imshow("bilinear interp", bilinear_result)
cv2.waitKey(0)
