import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'
'''
K.image_data_format() == 'channels_first': 检查Keras后端使用的图像数据格式是否为channels_first。
这种格式意味着图像数据的维度顺序为（通道数，高度，宽度）。然而，这行代码实际上没有设置格式，只是进行了检查。
'''
if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./test2.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print('the answer is: ',utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)
'''
img_nor = img_RGB/255: 将图像数据归一化到[0,1]范围。
img_nor = np.expand_dims(img_nor,axis = 0): 在图像数据的第0维增加一个维度，使其成为四维数组，以符合Keras模型输入的要求。
img_resize = utils.resize_image(img_nor,(224,224)): 使用utils模块中的resize_image函数将图像调整为224x224像素，这是AlexNet模型输入的期望尺寸。


使用Keras等框架时，模型通常期望输入数据具有特定的形状。对于图像分类任务，模型通常期望输入是一个四维数组，其形状为 (batch_size, height, width, channels)。这里：

batch_size：批次大小，表示一次输入模型的图像数量。
height：图像的高度。
width：图像的宽度。
channels：图像的通道数（例如，RGB图像有3个通道）。
当你处理单张图像时，图像数据通常是一个三维数组，形状为 (height, width, channels)。为了使这张图像符合模型的输入要求，你需要在第0维（即最前面的维度）增加一个维度，以表示批次大小为1。

为什么要写这一步：
符合模型输入要求：大多数深度学习框架（包括Keras）要求输入数据的批次维度，即使你只处理单张图像。通过增加一个维度，你将三维图像数组转换为四维数组，使其符合模型的输入形状要求。

统一数据格式：在处理多张图像时，通常会将它们堆叠成一个四维数组，其中第一维是批次大小。对于单张图像，增加一个维度可以保持数据格式的一致性，简化数据处理流程。

避免错误：如果不增加这个维度，直接将三维图像数组输入到模型中，可能会导致错误，因为模型期望的输入形状与实际输入形状不匹配。

通过执行 img_nor = np.expand_dims(img_nor, axis=0)，你将单张图像的三维数组转换为四维数组，形状变为 (1, height, width, channels)，这样就可以顺利地输入到模型中进行预测。
'''