import cv2
import numpy as np
from cv2 import destroyWindow
from matplotlib import pyplot as plt


def nearest_interp(img):
    height, width,channels = img.shape
    emptyimage = np.zeros((750,750,channels),np.uint8)

    for i in range(750):
        for j in range(750):
            emptyimage[i,j] = img[int(height*i/750 + 0.5),int(width*j/750 + 0.5)]
    return emptyimage

def bilinear_interp(img,out_dim):
    src_h,src_w,channels = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    dst_image = np.zeros((dst_h,dst_w,3),dtype = np.uint8)
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    for i in range(channels):
        for y in range(dst_h):
            for x in range(dst_w):
                x0 = (x + 0.5)*scale_x - 0.5
                y0 = (y + 0.5)*scale_y - 0.5
                x1 = int(np.floor(x0))
                y1 = int(np.floor(y0))
                x2 = min(x1 + 1,src_w -1)
                y2 = min(y1 + 1,src_h -1)
                temp_r0 = (x2 - x0)*img[y1,x1,i] + (x0 - x1)*img[y1,x2,i]
                temp_r1 = (x2 - x0)*img[y2,x1,i] + (x0 - x1)*img[y2,x2,i]
                dst_image[y,x,i] = int((y2 - y0)*temp_r0 + (y0 - y1)*temp_r1)
    return dst_image






if __name__ == '__main__':
    near = 0
    bi_linear = 1
    hist_gram = 0
    hist_enable = 1
    if near:
        img = cv2.imread('lenna.png')
        zoom = nearest_interp(img)
        cv2.imshow('zoom',zoom)
        cv2.imshow('original',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if bi_linear:
        img = cv2.imread('lenna.png')
        dst = bilinear_interp(img,(800,600))
        cv2.imshow('bilinear',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if hist_enable:
        img = cv2.imread("lenna.png", 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst_res = cv2.equalizeHist(gray)
        hist = cv2.calcHist([dst_res], [0], None, [256], [0, 256])
        plt.figure()
        plt.hist(dst_res.ravel(), 256)
        plt.show()
        cv2.imshow("Histogram Equalization", np.hstack([gray, dst_res]))
        cv2.waitKey(0)


