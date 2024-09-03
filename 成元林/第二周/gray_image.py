import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data,filters
from skimage.util import img_as_ubyte
#图片显示的方法
def imageshow(imgMatrix):
    # 根据维度判断显示，一维，二维直接显示灰度图，三维则需要转换通道，因为cv2是BGR通道
    if imgMatrix.ndim == 2 or imgMatrix.ndim == 1:
        plt.imshow(imgMatrix,cmap="gray")
    elif imgMatrix.ndim == 3:
        plt.imshow(imgMatrix,cv2.COLOR_BGR2RGB)

#采用点积或者循环方式实现灰度图或者二值图
def imageGrayOrigin(filepath):
    ori_img = cv2.imread(filepath)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    gray_img = np.dot(ori_img[...,:3],[0.3,0.59,0.11])
    return gray_img

#使用cv2实现灰度
def imageGrayByCv2(filepath):
    ori_img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2GRAY)
    return gray_img

#使用skiimage，实现灰度
def imageGrayBySki(filepath):
    ori_img = cv2.imread(filepath)
    ori_img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    gray_image = rgb2gray(ori_img)
    return gray_image


#图像二值化,传入灰度化后的图像矩阵和阈值
def imageBinary(imgMatrix):
    # 使用otsu查找阈值更加精细
    threshold = filters.threshold_otsu(imgMatrix)
    binary_img = np.where(imgMatrix>threshold,1,0)
    return binary_img

if __name__ == '__main__':
    # 原始方法灰度
    gray_image = imageGrayOrigin("lenna.png")
    # 画布2行3列，第一行第一列
    plt.subplot(3,2,1)
    #显示图片
    imageshow(gray_image)
    #画布2行三列，第二列
    plt.subplot(3, 2, 2)
    #二值化
    binary_image = imageBinary(gray_image)
    print("origin:%s"%binary_image)
    imageshow(binary_image)

    #opencv方式
    gray_image2 = imageGrayByCv2("lenna.png")
    plt.subplot(3,2,3)
    imageshow(gray_image2)
    binary_image2 = imageBinary(gray_image2)
    plt.subplot(3, 2, 4)
    print("cv2:%s" % binary_image2)
    imageshow(binary_image2)

    gray_image3 = imageGrayBySki("lenna.png")
    plt.subplot(3, 2, 5)
    imageshow(gray_image3)
    # 使用Otsu方法找到阈值
    binary_image3 = imageBinary(gray_image3)
    plt.subplot(3, 2, 6)
    print("binary_image3:%s" % binary_image3)
    imageshow(binary_image3)

    plt.show()




