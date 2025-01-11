import cv2
import numpy as np

def method(img,new_h,new_w):
    h, w, c = img.shape
    tar_h, tar_w = new_h, new_w
    if h == tar_h and w == tar_w:
        return img.copy()
    #空目标图
    tar_img = np.zeros((tar_h,tar_w,c),np.uint8)

    #缩放比例
    s_x, s_y = float(h)/new_h, float(w)/new_w
    for i in range(c):
        for tar_x in range(tar_h):
            for tar_y in range(tar_w):
                #目标图在原图上所对应的坐标
                x = (tar_x + 0.5) * s_x - 0.5
                y = (tar_y + 0.5) * s_y - 0.5

                x1 = int(np.floor(x))#向下取整
                x2 = min(x1 + 1, w - 1)#防止超出边界
                y1 = int(np.floor(y))
                y2 = min(y1 + 1, h - 1)

                temp0 = (x2 - x)*img[x1,y1,i] + (x - x1)*img[x1,y2,i]
                temp1 = (x2 - x)*img[x1,y2,i] + (x - x1)*img[x2,y2,i]

                tar_img[tar_x, tar_y, i] = int((y2 - y)*temp0 + (y - y1)*temp1)
    return tar_img


if __name__ == '__main__':
    img = cv2.imread('girls.jpg')
    cv2.imshow('img',img)

    img1 = method(img, 600, 600)
    cv2.imshow('img1', img1)
    cv2.waitKey(0)
