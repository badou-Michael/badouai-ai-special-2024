#-*- coding:utf-8 -*-
import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('img/timg.jpg')
model = mtcnn()
threshold = [0.5, 0.6, 0.7]
rectangles = model.detectFace(img, threshold)
img_out = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        width = int(rectangle[2] - rectangle[0])
        height = int(rectangle[3] - rectangle[1])
        paddingH = 0.01 * width
        paddingW = 0.02 * height
        crop_img = img[int(rectangle[1]+paddingH):rectangle[3]-paddingH,
                       int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
        if crop_img is None:
            continue
        elif crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectange(img_out, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2], int(rectangle[3])), (255,0,0), 1))
        for i in range(5, 15, 2): # 5,7,9,11,13 -- 人脸特征点总共5个
            cv2.circle(img_out, (int(rectangle[i]), int(rectangle[i+1])), 2, (0,255,0)) # 绿色, 半径2

cv2.imwrite('img/out.jpg', img_out)
cv2.imshow("out", img_out)
cv2.waitKey(0)