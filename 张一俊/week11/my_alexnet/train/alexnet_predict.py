# 根据保存下来的训练结果进行推理

import numpy as np
from my_alexnet.train import utils
import cv2
from keras import backend as K
from my_alexnet.model.alexnet import AlexNet

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5", by_name=True)
    img = cv2.imread("./predict_data/test2.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (227, 227))
    # utils.print_answer(np.argmax(model.predict(img)))
    print('the answer is: ', utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo", img)
    cv2.waitKey(0)

# pip install tensorflow==1.15.0 keras==2.3.1 不管用
# pip3 install h5py==2.10.0 这个可以
