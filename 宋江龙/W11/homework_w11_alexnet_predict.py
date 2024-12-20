#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/12 23:18
@Author  : Mr.Long
"""

import numpy as np
import cv2
from keras import backend as bk
from src.practiceModule.study_ai.homework.homework_w11_alexnet_train import resize_image_w11, print_answer_w11, alexnet_w11

# K.set_image_dim_ordering('tf')
bk.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model_w11 = alexnet_w11()
    model_w11.load_weights("D:\workspace\data\logs\\last1.h5")
    img = cv2.imread("D:\workspace\data\\alexnet_data\\test2.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = resize_image_w11(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print('the answer is: ', print_answer_w11(np.argmax(model_w11.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)
