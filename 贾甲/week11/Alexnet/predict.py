#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-12-12
import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet


K.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("D:/01 贾甲/04八斗/week11/alexnet/AlexNet-Keras-master/logs/last1.h5")
    img = cv2.imread("D:/01 贾甲/04八斗/week11/alexnet/AlexNet-Keras-master/test2.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))

    print('the answer is: ',utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)
