import cv2
import numpy as np
from matplotlib import pyplot as plt

#浮点计算
def RGBImage2GreyImage(img):
    #宽、高、通道
    w, h, c = img.shape
    greyImage = img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.30
    return greyImage

def RGBImage2BinaryImage(img):
    w, h, c = img.shape
    greyImage = img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.30
    greyImage /= 255
    binaryImage = np.where(greyImage > 0.5, 1, 0)
    return binaryImage


if __name__ == '__main__':
    imagePath = "lenna.png"
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    greyImage = RGBImage2GreyImage(img)
    binaryImage = RGBImage2BinaryImage(img)

    cv2.imwrite(imagePath.split('.')[0] + "Binary.png", binaryImage * 255)
    cv2.imwrite(imagePath.split('.')[0] + "Grey.png", greyImage)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    # 原图
    axes[0].imshow(img)
    axes[0].axis('off')
    # 灰度图
    axes[1].imshow(greyImage, cmap='gray')
    axes[1].axis('off')
    # 二值图
    axes[2].imshow(binaryImage, cmap='gray')
    axes[2].axis('off')
    plt.show()
