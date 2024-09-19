import cv2
import numpy as np

def bilinear_interpolation(img, dst_img):
    width, height, channels = img.shape
    # if dst_img.shape == img.shape:
    #     return img
    scale_x = width/dst_img[0]
    scale_y = height/dst_img[1]
    new_img = np.zeros((dst_img[0], dst_img[1], channels), np.uint8)
    for c in range(channels):
        for i in range(dst_img[0]):
            for j in range(dst_img[1]):
# 图像中心化
                x = (i + 0.5) * scale_x - 0.5
                y = (j + 0.5) * scale_y - 0.5

                x0 = int(x)
                x1 = min(x0 + 1, width - 1)
                y0 = int(y)
                y1 = min(y0 + 1, height - 1)

                temp0 = (x1 - x) * img[x0, y1, c] + (x - x0) * img[x1, y1, c]
                temp1 = (x1 - x) * img[x0, y0, c] + (x - x0) * img[x1, y0, c]
                new_img[i, j, c] = int((y1 - y) * temp1 + (y - y0) * temp0)

    return new_img


img = cv2.imread("lenna.png")
bin_img = bilinear_interpolation(img, (800, 800))
cv2.imshow("oringal img", img)
cv2.imshow("bilinear img", bin_img)
cv2.waitKey(0)


