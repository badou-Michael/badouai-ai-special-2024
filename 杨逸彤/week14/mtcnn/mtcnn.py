import cv2
import numpy as np
from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model

import utils

# 第一阶段：P-NET->全卷积网络
def create_Pnet(weight_path):
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

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 第二阶段：R-Net的目的是为了去除大量的非人脸框，非极大值抑制
def create_Rnet(weight_path):
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

# 第三阶段：Onet比R-Net层又多了一层卷积层
def create_Onet(weight_path):
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
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # 归一化
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape

        scales = utils.calculateScales(img)

        out = []
        #pnet：第一层P-Net将经过卷积，池化操作后输出分类（对应像素点是否存在人脸）和回归（回归 box)结果
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        image_num = len(scales)
        faces = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:,:,1]
            roi = out[i][1][0]

            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            faces.extend(rectangle)

        faces = utils.NMS(faces, 0.7)

        if len(faces) == 0:
            return faces

        # rnet：第二层网络将第一层输出的结果使用非极大抑制（NMS）来去除高度重合的候选框，并将这些候选框放入R-Net中进行精细的操作，
        # 拒绝大量错误框，再对回归框做校正，并使用NMS去除重合框，输出分支同样两个分类和回归
        predict_24_batch = []
        for face in faces:
            crop_img = copy_img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0] # 人脸概率
        cls_prob = np.array(cls_prob) # 框位置
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        faces = utils.filter_face_24net(cls_prob, roi_prob, faces, origin_w, origin_h, threshold[1])

        if len(faces) == 0:
            return faces

        # onet： 最后将R-Net输出认为是人脸的候选框输入到O-Net中再一次进行精细操作，拒绝掉错误的框， 此时输出分支包含三个分类：
        # a. 是否有人脸：2个输出；
        # b. 回归：回归得到的框的起始点（或中心点）的xy坐标和框的长宽，4个输出；
        # c. 人脸特征点定位：5个人脸特征点的xy坐标，10个输出。
        predict_batch = []
        for face in faces:
            crop_img = copy_img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        faces = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, faces, origin_w, origin_h, threshold[2])

        return faces
