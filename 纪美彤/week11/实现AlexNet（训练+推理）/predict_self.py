import numpy as np
import utils
import cv2
from keras import backend as K
from ALEXNET_SELF import AlexNet_SELF

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model = AlexNet_SELF()
    # 加载模型
    model.load_weights("./logs/last1.h5")
    # 读取图片并处理成网络输入形式
    img = cv2.imread("./test2.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print('the answer is: ',utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)
