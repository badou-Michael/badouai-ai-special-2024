from keras.layers import Conv2D, Input, MaxPool2D, Permute, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import numpy as np
import utils
import cv2

def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])
    x = Conv2D(10, (3,3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3,3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2], name='PReLU2')(x)

    x = Conv2D(32, (3,3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2], name='PReLU3')(x)

    classifier = Conv2D(2, (1,1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4,(1,1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3,3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 11,11,28 -> 4,4,48 (已验证)
    x = Conv2D(48, (3,3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2,2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2], name='prelu3')
    # 3,3,64 -> 64,3,3
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)
    # 128 -> 2, 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Onet(weight_path):
    input = Input(shape=[48,48,3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3,3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 10,10,64 -> 4,4,64
    x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2) # strides=2
    # 4.4.64 -> 3,3,128
    x = Conv2D(128, (2,2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2], name='prelu4')(x)
    # 3,3,128 -> 128,3,3
    x = Permute((3,2,1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name = 'conv5')(x)
    x = PReLU(name = 'prelu5')(x)

    # 256 -> 2,4,10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)
    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('./model_data/pnet.h5')
        self.Rnet = create_Rnet('./model_data/rnet.h5')
        self.Onet = create_Onet('./model_data/onet.h5')

    def detectFace(self, img, threshold): # 重要
        # 归一化, 加快收敛速度
        copy_img = (img.copy() - 127.5)/127.5
        origin_h, origin_w, _ = copy_img.shape
        # 获得每次缩放的比例
        scales = utils.calculateScales(img)

        out = []
        # 1. Pnet
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            # reshape(1, hs, ws, c) 适配深度学习输入需求
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率 out[i][0][0].shape = 245,400 (仅限i=0)
            cls_prob = out[i][0][0][:,:,1] # 245,400  仅提取1的位置, 忽略0
            # 其对应框的位置
            roi = out[i][1][0] # 245,400

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape # 245, 400 (仅限i=0, reshape过程中会变)
            out_side = max(out_h, out_w)
            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1/scales[i], 
                                                origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle) # 将rectangle的每个元素逐个添加到 rectangles
        # 进行非极大值抑制
        rectangles = utils.NMS(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles

        # 2. Rnet
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24,24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch) # 987,24,24,3
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = np.array(out[0])
        roi_prob = np.array(out[1])
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles,
                                             origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        # 3. Onet
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48,48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, 
                                             rectangles, origin_w, origin_h, threshold[2])
        return rectangles