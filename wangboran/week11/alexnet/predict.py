#-*- coding:utf-8 -*-
# author: 王博然
import numpy as np
import cv2
from keras import backend as K
import AlexNet

synset = {
    0: '猫',
    1: '狗'
}

if __name__ == '__main__':
    # print("Current image data format:", K.image_data_format()) # channels_last
    model = AlexNet.AlexNet()
    # 推理时 model.compile 不是必须的
    model.load_weights("./last1.h5")
    img = cv2.imread("./test2.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_RGB, (224, 224))
    img_nor = np.expand_dims(img_resize/255, axis = 0)
    print('answer: ', synset[np.argmax(model.predict(img_nor))])