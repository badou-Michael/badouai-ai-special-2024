import cv2
import numpy as np
def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)       # 要将原图像放大为800*800
    # emptyImage = np.zeros((800, 800, channels), img.dtype)
    # print(emptyImage.shape)
    # print(emptyImage)
    # 计算目标图像与原图像的比例关系
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)     # 由于int是向下取整，因此要+0.5来保证插值所取的像素值满足最邻近插值的原理
            y = int(j/sw + 0.5)     # 若j/sw=35.6，未加0.5时int(j)=35→不满足最邻近插值的原理（int(j)应取36）
            # 利用比例关系找到目标图像点（i，j）在原图像的对应位置（x，y），再将（x，y）的像素值赋给（i，j）
            emptyImage[i, j] = img[x, y]        # 若（x，y）超出原图像的范围（512,512），则程序会发生报错
    return emptyImage
    
# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


