import cv2
import numpy as np


def newsize(img, new_height, new_width):
    """最临近插值"""
    height, width, channels = img.shape
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    scale_h, scale_w = height / new_height, width / new_width

    for h in range(new_height):
        for w in range(new_width):
            des_h = min(int(h * scale_h + 0.5), height - 1)
            des_w = min(int(w * scale_w + 0.5), width - 1)
            new_img[h][w] = img[des_h, des_w]

    return new_img


img = cv2.imread("practice/cv/week03/lenna.png")
new_img = newsize(img, 800, 800)
#new_img=cv2.resize(img,(800,800),interpolation=cv2.INTER_NEAREST)
print(new_img)
print(new_img.shape)
cv2.imshow("nearest interp", new_img)
cv2.imshow("image", img)
cv2.waitKey(0)
