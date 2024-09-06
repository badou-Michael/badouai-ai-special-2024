from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def raw_2_binary(path):
    # 读取图像
    raw_img = cv2.imread(path)
    # BGR转RGB
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # RGB to Gray
    img_gray = rgb2gray(img)
    # Gray to binary image
    img_binary = np.where(img_gray >= 0.5, 1, 0)

    plt.imshow(img_binary, cmap='gray')
    plt.show()


if __name__ == '__main__':
    raw_2_binary('./lenna.png')
