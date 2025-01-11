'''
最近邻近插值
'''
import cv2
import numpy as np

def func(img, height, weight):
    img_height, img_weight, channel = img.shape[:3]
    empty_image = np.zeros((height, weight,channel), dtype=img.dtype)
    # 求出放大或缩小前后行列的比例
    height_pro = (height / img_height);
    weight_pro = (weight / img_weight);
    for i in range(height):
        for j in range(weight):
            x = int(i / height_pro + 0.5)
            y = int(j / weight_pro + 0.5)
            empty_image[i, j] = img[x, y]
    return empty_image

pth = "lenna.png"
image = cv2.imread(pth)
zoom = func(image, 400, 200)
print(zoom.shape)
cv2.imshow("Image", image)
cv2.imshow("Magnify", zoom)
cv2.waitKey(0)
