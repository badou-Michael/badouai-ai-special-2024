# -*- coding: utf-8 -*-
# time: 2024/11/26 23:19
# file: mtcnn.py
# author: flame
from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
import numpy as np
import utils
import cv2

''' 创建Pnet模型并加载权重。输入为任意大小的图像，输出为分类器和边界框回归器。 '''
def create_Pnet(weights_path):
    ''' 定义输入层，接受任意大小的RGB图像。 '''
    input = Input(shape=[None, None, 3])

    ''' 使用Conv2D层进行卷积操作，输出通道数为10，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为2x2。 '''
    x = MaxPool2D(pool_size=2)(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为16，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为32，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为2，卷积核大小为1x1，使用softmax激活函数进行分类。 '''
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    ''' 使用Conv2D层进行卷积操作，输出通道数为4，卷积核大小为1x1，用于边界框回归。 '''
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    ''' 定义模型，输入为图像，输出为分类器和边界框回归器。 '''
    model = Model([input], [classifier, bbox_regress])
    ''' 加载预训练权重。 '''
    model.load_weights(weights_path, by_name=True)
    ''' 返回构建好的模型。 '''
    return model

''' 创建Rnet模型并加载权重。输入为24x24的图像，输出为分类器和边界框回归器。 '''
def create_Rnet(weights_path):
    ''' 定义输入层，接受24x24的RGB图像。 '''
    input = Input(shape=[24, 24, 3])
    ''' 使用Conv2D层进行卷积操作，输出通道数为28，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为3x3，步长为2，使用填充。 '''
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为48，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为3x3，步长为2。 '''
    x = MaxPool2D(pool_size=3, strides=2)(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为64，卷积核大小为2x2，步长为1，不使用填充。 '''
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    ''' 使用Permute层重新排列张量的维度。 '''
    x = Permute((3, 2, 1))(x)
    ''' 使用Flatten层将多维张量展平为一维向量。 '''
    x = Flatten()(x)

    ''' 使用Dense层进行全连接操作，输出节点数为128。 '''
    x = Dense(128, name='conv4')(x)
    ''' 使用PReLU激活函数。 '''
    x = PReLU(name='prelu4')(x)

    ''' 使用Dense层进行全连接操作，输出节点数为2，使用softmax激活函数进行分类。 '''
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    ''' 使用Dense层进行全连接操作，输出节点数为4，用于边界框回归。 '''
    bbox_regress = Dense(4, name='conv5-2')(x)

    ''' 定义模型，输入为图像，输出为分类器和边界框回归器。 '''
    model = Model([input], [classifier, bbox_regress])
    ''' 加载预训练权重。 '''
    model.load_weights(weights_path, by_name=True)
    ''' 返回构建好的模型。 '''
    return model

''' 创建Onet模型并加载权重。输入为48x48的图像，输出为分类器、边界框回归器和关键点回归器。 '''
def create_Onet(weights_path):
    ''' 定义输入层，接受48x48的RGB图像。 '''
    input = Input(shape=[48, 48, 3])

    ''' 使用Conv2D层进行卷积操作，输出通道数为32，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为3x3，步长为2，使用填充。 '''
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为64，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为3x3，步长为2。 '''
    x = MaxPool2D(pool_size=3, strides=2)(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为64，卷积核大小为3x3，步长为1，不使用填充。 '''
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    ''' 使用MaxPool2D层进行最大池化操作，池化窗口大小为2x2。 '''
    x = MaxPool2D(pool_size=2)(x)

    ''' 使用Conv2D层进行卷积操作，输出通道数为128，卷积核大小为2x2，步长为1，不使用填充。 '''
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    ''' 使用PReLU激活函数，共享轴为1和2，即在每个特征图上共享参数。 '''
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    ''' 使用Permute层重新排列张量的维度。 '''
    x = Permute((3, 2, 1))(x)
    ''' 使用Flatten层将多维张量展平为一维向量。 '''
    x = Flatten()(x)
    ''' 使用Dense层进行全连接操作，输出节点数为256。 '''
    x = Dense(256, name='conv5')(x)
    ''' 使用PReLU激活函数。 '''
    x = PReLU(name='prelu5')(x)

    ''' 使用Dense层进行全连接操作，输出节点数为2，使用softmax激活函数进行分类。 '''
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    ''' 使用Dense层进行全连接操作，输出节点数为4，用于边界框回归。 '''
    bbox_regress = Dense(4, name='conv6-2')(x)
    ''' 使用Dense层进行全连接操作，输出节点数为10，用于关键点回归。 '''
    landmark_regress = Dense(10, name='conv6-3')(x)

    ''' 定义模型，输入为图像，输出为分类器、边界框回归器和关键点回归器。 '''
    model = Model([input], [classifier, bbox_regress, landmark_regress])
    ''' 加载预训练权重。 '''
    model.load_weights(weights_path, by_name=True)
    ''' 返回构建好的模型。 '''
    return model

class mtcnn():
    ''' 初始化mtcnn类，加载Pnet、Rnet和Onet模型。 '''
    def __init__(self):
        ''' 创建Pnet模型并加载权重。 '''
        self.Pnet = create_Pnet('model_data/pnet.h5')
        ''' 创建Rnet模型并加载权重。 '''
        self.Rnet = create_Rnet('model_data/rnet.h5')
        ''' 创建Onet模型并加载权重。 '''
        self.Onet = create_Onet('model_data/onet.h5')

    ''' 检测图像中的人脸。 '''
    def detectFace(self, img, threshold):
        ''' 复制图像并进行归一化处理。 '''
        copy_img = (img.copy() - 127.5) / 127.5
        ''' 获取图像的原始高度和宽度。 '''
        origin_h, origin_w, _ = copy_img.shape
        ''' 计算不同尺度下的图像。 '''
        scales = utils.calculateScales(img)

        ''' 存储Pnet模型的输出结果。 '''
        out = []
        ''' 遍历每个尺度的图像。 '''
        for scale in scales:
            ''' 计算缩放后的高度和宽度。 '''
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            ''' 缩放图像。 '''
            scale_img = cv2.resize(copy_img, (ws, hs))
            ''' 将图像重塑为模型输入的格式。 '''
            inputs = scale_img.reshape(1, *scale_img.shape)
            ''' 使用Pnet模型预测。 '''
            output = self.Pnet.predict(inputs)
            ''' 将预测结果存储到out列表中。 '''
            out.append(output)

        ''' 计算不同尺度下的矩形框。 '''
        image_num = len(scales)
        rectangles = []
        ''' 遍历每个尺度的预测结果。 '''
        for i in range(image_num):
            ''' 获取分类概率图。 '''
            cls_prob = out[i][0][0][:, :, 1]
            ''' 获取边界框回归图。 '''
            bbox_reg = out[i][1][0]
            ''' 获取分类概率图的高度和宽度。 '''
            out_h, out_w = cls_prob.shape
            ''' 获取输出的最大边长。 '''
            out_side = max(out_h, out_w)
            ''' 打印分类概率图的形状。 '''
            print("矩阵的形状：", cls_prob.shape)
            ''' 使用utils模块检测12网的人脸。 '''
            rectangle = utils.detect_face_12net(cls_prob, bbox_reg, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            ''' 将检测到的矩形框添加到rectangles列表中。 '''
            rectangles.extend(rectangle)

        ''' 使用非极大值抑制（NMS）过滤矩形框。 '''
        rectangles = utils.NMS(rectangles, 0.7)

        ''' 如果没有检测到矩形框，直接返回。 '''
        if len(rectangles) == 0:
            return rectangles

        ''' 存储Rnet模型的输入图像。 '''
        predict_24_batch = []
        ''' 遍历每个矩形框。 '''
        for rectangle in rectangles:
            ''' 裁剪矩形框内的图像。 '''
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            ''' 缩放图像到24x24。 '''
            crop_img = cv2.resize(crop_img, (24, 24))
            ''' 将裁剪后的图像添加到predict_24_batch列表中。 '''
            predict_24_batch.append(crop_img)

        ''' 将裁剪后的图像转换为数组。 '''
        predict_24_batch = np.array(predict_24_batch)
        ''' 使用Rnet模型预测。 '''
        out = self.Rnet.predict(predict_24_batch)

        ''' 获取分类概率。 '''
        cls_prob = out[0]
        ''' 将分类概率转换为数组。 '''
        cls_prob = np.array(cls_prob)
        ''' 获取边界框回归结果。 '''
        roi_prob = out[1]
        ''' 使用utils模块过滤24网的人脸。 '''
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        ''' 如果没有检测到矩形框，直接返回。 '''
        if len(rectangles) == 0:
            return rectangles

        ''' 存储Onet模型的输入图像。 '''
        predict_batch = []
        ''' 遍历每个矩形框。 '''
        for rectangle in rectangles:
            ''' 裁剪矩形框内的图像。 '''
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            ''' 缩放图像到48x48。 '''
            scale_img = cv2.resize(crop_img, (48, 48))
            ''' 将裁剪后的图像添加到predict_batch列表中。 '''
            predict_batch.append(scale_img)

        ''' 将裁剪后的图像转换为数组。 '''
        predict_batch = np.array(predict_batch)
        ''' 使用Onet模型预测。 '''
        out = self.Onet.predict(predict_batch)
        ''' 获取分类概率。 '''
        cls_prob = out[0]
        ''' 获取边界框回归结果。 '''
        bbox_reg = out[1]
        ''' 获取关键点回归结果。 '''
        landmark_reg = out[2]
        ''' 使用utils模块过滤48网的人脸。 '''
        rectangles = utils.filter_face_48net(cls_prob, bbox_reg, landmark_reg, rectangles, origin_w, origin_h, threshold[2])
        ''' 返回最终检测到的矩形框。 '''
        return rectangles

