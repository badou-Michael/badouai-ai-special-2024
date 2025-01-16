#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：detect.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 22:18 
@Desc : 
"""
import cv2
from model import MTCNN

def main():

    img = cv2.imread("img/timg.png")

    model = MTCNN()
    threshold = [0.5, 0.6, 0.7]
    rectangles = model.detect_face(img, threshold)
    draw = img.copy()
    for rectangle in rectangles:
        if rectangle is not None:
            width = -int(rectangle[0]) + int(rectangle[2])
            height = -int(rectangle[1]) + int(rectangle[3])
            padding_height = 0.01 * width
            padding_width = 0.02 * height
            crop_img = img[int(rectangle[1] + padding_width):int(rectangle[3] - padding_height),
                       int(rectangle[0] - padding_height):int(rectangle[2] + padding_width)]
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 1)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))


    cv2.imwrite("img/out.png", draw)
    img.show("test", draw)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
