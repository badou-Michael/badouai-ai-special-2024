import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

K.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/AlexNetLastModel.h5")
    img = cv2.imread("./test1.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255 #图像标准化
    img_nor = np.expand_dims(img_nor,axis=0)
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    # 模型评估并输出结果
    print('the answer is: ',utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("figure",img)
    cv2.waitKey(0)
