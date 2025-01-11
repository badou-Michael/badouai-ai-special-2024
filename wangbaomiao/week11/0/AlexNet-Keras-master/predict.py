# -*- coding: utf-8 -*-
# time: 2024/11/19 17:21
# file: predict.py
# author: flame
import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

''' 整体逻辑：设置图像数据格式为 'channels_last'，加载预训练的 AlexNet 模型，读取并处理测试图像，预测图像类别并打印结果，最后显示图像。 '''

''' 设置图像数据格式，'channels_last' 表示最后一个维度用于表示颜色通道，这通常用于 TensorFlow 后端。 '''
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    ''' 创建 AlexNet 模型实例。 '''
    model = AlexNet()

    ''' 加载预训练的模型权重文件，路径为 './model/AlexNet_weights.h5'。 '''
    model.load_weights('./model/AlexNet_weights.h5')

    ''' 读取测试图像文件，路径为 './Test.jpg'。 '''
    img = cv2.imread('./Test.jpg')

    ''' 将图像从 BGR 格式转换为 RGB 格式，因为 OpenCV 默认读取图像为 BGR 格式。 '''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ''' 将图像归一化到 [0, 1] 范围，通过除以 255。 '''
    img_nor = img_RGB / 255

    ''' 将图像调整大小为 224x224 像素，这是 AlexNet 模型输入所需的尺寸。 '''
    img_resize = utils.resize_iamge(img_nor, [224, 224])

    ''' 使用模型预测图像类别，并获取预测结果中概率最高的类别索引。 '''
    ''' 打印预测结果对应的类别名称。 '''
    print(utils.print_answer(np.argmax(model.predict(img_resize))))

    ''' 显示处理后的图像，窗口标题为 'img'。 '''
    cv2.imshow('img', img)

    ''' 等待用户按键事件，0 表示无限等待。 '''
    cv2.waitKey(0)
