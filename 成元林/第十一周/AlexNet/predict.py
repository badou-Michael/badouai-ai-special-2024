import numpy as np

import cv2

from keras import backend as K

from 成元林.第十一周.AlexNet import utils
from 成元林.第十一周.AlexNet.model.AlexModelOfKeras import AlexNet_cat_dog_Model

K.set_image_data_format('channels_last') #返回的值中最后的值为通道数

if __name__ == "__main__":
    model = AlexNet_cat_dog_Model() #返回AlexNet的模型
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("dog_local.jpg") #img为需要验证的猫或狗的图片
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #将img转为RGB模式
    img_nor = img_RGB/255 #此时为（658，526，3）的数据，将像素值降到0-1之见
    img_nor = np.expand_dims(img_nor,axis = 0) #用于扩展数组形状，在0的维数处增加一个维度，变为（1，488，500，3）的数据
    img_resize = utils.resize_image(img_nor,(224,224)) #改变数组大小为（224，224）
    arr = model.predict(img_resize) #返回的值为得出的每一种类别的概率，predict函数为karas中自带
    print(arr) #输出为以一维数组，有两个值，分别为是猫的概率和是狗的概率
    if(arr[0][0]>0.9):
        print(utils.print_answer(np.argmax(arr)))
        #argmax返回数组中最大值的索引
    elif(arr[0][1]>0.9):
        print(utils.print_answer(np.argmax(arr)))

    cv2.imshow("test_img",img)
    cv2.waitKey(0)
