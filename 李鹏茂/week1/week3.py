import cv2
import numpy as np
import matplotlib.pyplot as plt

def dancha(img):
    height,width,tongdao= img.shape
    print(height,width,tongdao)
    zero_jvzhen = np.zeros((int(height*1.5),int(width*1.5),tongdao),np.uint8)
    for i in range(int(height*1.5)):
        for j in range(int(width*1.5)):
            x=int(i/1.5 +0.5)
            y= int(j/1.5 +0.5)
            zero_jvzhen[i][j]=img[x,y]
    return zero_jvzhen



if __name__ == '__main__':
    image = cv2.imread(r'C:\Users\Lenovo\Desktop\meinv.png')
    # 计算新尺寸
    new_size = (int(image.shape[1] * 1.5), int(image.shape[0] * 1.5))

    # 单线性插值
    resized_linear = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    imgdan=dancha(image)
    # 双线性插值
    resized_bilinear = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    # 保存结果
    cv2.imshow('dan', resized_linear)
    cv2.imshow('shuang', resized_bilinear)
    cv2.imshow('linjin', imgdan)
    cv2.waitKey()