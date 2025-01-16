from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2


# 3次卷积和1次池化操作后，原来12*12*3的矩阵变为1*1*32
# • 利用这个1*1*32的向量，再通过一个1*1*2的卷积，得到了”是否是人脸”的分类结果
# • 我们令输入图片矩阵为A，卷积核在原图矩阵A上滑动，把每个12*12*3区域的矩阵都计算成该区域有
# 无人脸的得分，最后可以得到一个二维矩阵为S，S每个元素的值是[0, 1]的数，代表有人脸的概率。即
# A通过一系列矩阵运算，变化到S。
def Pnet_self(weight_path):
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)

    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
    # facial_landmark在此处没有输出，所以省略
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 该网络结构只是和P-Net网络结构多了一个全连接层。
# 图片在输入R-Net之前，都需要缩放到24x24x3。网络的输出与P-Net是相同的，R-Net的目的是为
# 了去除大量的非人脸框。
def Rnet_self(weight_path):
    input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 该层比R-Net层又多了一层卷积层，所以处理的结果会更加精细。输入的图像大小48x48x3，输出包
# 括N个边界框的坐标信息，score以及关键点位置。
def Onet_self(weight_path):
    input = Input(shape = [48,48,3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = Permute((3,2,1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model

class mtcnn():
    def __init__(self):
        self.Pnet = Pnet_self('model_data/pnet.h5')
        self.Rnet = Rnet_self('model_data/rnet.h5')
        self.Onet = Onet_self('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # 图像归一化，将0-255归一到-1~1
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        #-----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        #-----------------------------#
        scales = utils.calculateScales(img)

        out = []
        # 将归一化后的原图分别缩小不同的尺度
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            # 将所有output加入out
            out.append(output)

        image_num = len(scales)
        rectangles = []

        # P-Net的输出:
        # 1. 网络的第一部分输出是用来判断该图像是否存在人脸，输出向量大小1x1x2，也就是两个值。
        # 2. 网络的第二部分给出框的精确位置，即边框回归：P-Net输入的12×12的图像块可能并不是完美的
        # 人脸框的位置，如有的时候人脸并不正好为方形，有可能12×12的图像偏左或偏右，因此需要输出
        # 当前框位置相对完美的人脸框位置的偏移。这个偏移大小为1×1×4，即表示框左上角的横坐标的相
        # 对偏移，框左上角的纵坐标的相对偏移、框的宽度的误差、框的高度的误差。
        # 3. 网络的第三部分给出人脸的5个关键点的位置。5个关键点分别对应着左眼的位置、右眼的位置、
        # 鼻子的位置、左嘴巴的位置、右嘴巴的位置。每个关键点需要两维来表示，因此输出是向量大小
        # 为1×1×10。
        for i in range(image_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:,:,1]
            # 其对应的框的位置
            roi = out[i][1][0]

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        # 进行非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # 进入Rnet之前,将Pnet识别出的框裁切并resize为24*24
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles


        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles
