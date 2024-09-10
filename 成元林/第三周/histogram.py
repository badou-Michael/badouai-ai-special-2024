import cv2
import numpy as np
import matplotlib.pyplot as plt
def histogram(filepath):
    # 读取图像
    oriimg = cv_imread(filepath)
    # 分割图像为3通道
    (b,g,r) = cv2.split(oriimg)
    bH = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    mergeimg = cv2.merge((bH, gh, rh))
    return mergeimg

# 显示图像的直方图
def showhist(filepath):
    imread_img = cv_imread(filepath)
    colors = ("b","g","r")
    arr = cv2.split(imread_img)
    for item,c in zip(arr,colors):
        hist = cv2.calcHist([item], [0], None, [256], [0, 256])
        print(hist)
        plt.plot(hist,color=c)
        # plt.hist()
    plt.figure(figsize=(10,7))
    plt.show()

# 中文路劲转换后读取
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

if __name__ == '__main__':
    # cv2.imread("../sss");

    # 展示直方图
    showhist("../第二周/lenna.png")
    # 彩色直方图均衡化
    # hisImg = histogram("../第二周/lenna.png")
    # cv2.imshow("dst",hisImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()