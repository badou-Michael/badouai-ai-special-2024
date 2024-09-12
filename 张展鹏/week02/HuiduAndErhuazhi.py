import cv2
import matplotlib.pyplot as plt
import  numpy as np

#最邻近插值
def chaZhi(img):
    h,w,c = img.shape
    czImg = np.zeros((640,1500,c),dtype=np.uint8)
    sh = 640/h
    sw = 1500/w
    for i in range(640):
        for j in range(1500):
            a = int(i/sh + 0.5)  #主要解决四舍五入的问题，int是向下取整的，当明确了是最邻近插值方法就需要这样，否则根据业务需求
            b = int(j/sw + 0.5)
            czImg[i,j] = img[a,b]

    return czImg



if __name__ =='__main__':

    imghdr = cv2.imread('../lbxx.jpg')

    # czImg = cv2.resize(imghdr,(640,1500,3),near)  #near代表最邻近，bin代表双线性

    print("---------------------------原图---------------------------")
    print(imghdr)
    czImg = chaZhi(imghdr)
    print("---------------------------缩小后的图---------------------------")
    print(czImg)
    print("---------------------------原图shape---------------------------")
    print(imghdr.shape)
    print("---------------------------缩小后的图shape---------------------------")
    print(czImg.shape)
    cv2.imshow('原图',imghdr)
    cv2.imshow('缩放后的图',czImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
