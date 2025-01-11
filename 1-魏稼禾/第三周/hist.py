import cv2
import numpy as np
from matplotlib import pyplot as plt

#手动实现equalizeHist
def diy_equalizeHist(gray):
    H,W = gray.shape
    Hist = np.zeros(256, np.int32)
    for h in range(H):
        for w in range(W):
            Hist[gray[h,w]] += 1
    Hist_acc = np.zeros(256, np.int32)
    acc = 0
    for i in range(256):
        Hist_acc[i] = Hist[i] + acc
        acc = Hist_acc[i]
    dst = np.zeros((H,W), np.uint8)
    rate = float(256)/(H*W)
    for h in range(H):
        for w in range(W):
            dst[h,w] = int(Hist_acc[gray[h,w]] * rate - 1)
    return dst

#实现多通道均衡化
def mulity_channels_eh(img):
    chans = cv2.split(img)
    (b,g,r) = chans
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    res = cv2.merge((bH,gH,rH))
    return res

if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    dst_diy = diy_equalizeHist(gray)
    dst_chans = mulity_channels_eh(img)
    cv2.imshow("equalizeHist",dst)
    cv2.imshow("equalizeHist_diy",dst_diy)
    cv2.imshow("mulity_chans_eh",dst_chans)
    cv2.waitKey()
    cv2.destroyAllWindows()