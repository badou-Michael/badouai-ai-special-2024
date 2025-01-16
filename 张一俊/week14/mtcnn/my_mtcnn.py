from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import numpy as np
import cv2
from mtcnn import utils


# 第一段，粗略获取人脸框
def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# 第二段, 精修
def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])

    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

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
    x = PReLU(name='prelu4')(x)

    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


# 第三段，获得五个点
def create_Onet(weight_path):
    input = Input(shape=[48, 48, 3])

    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model


class MTCNN:
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # 归一化，加快收敛速度
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape


        # 计算图像金字塔
        scales = utils.calculateScales(img)
        out = []

        # PNet处理
        for scale in scales:
            hs, ws = int(origin_h * scale), int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        rectangles = self._process_pnet_output(out, scales, origin_w, origin_h, threshold[0])

        if len(rectangles) == 0:
            return rectangles

        # RNet处理
        rectangles = self._process_rnet_output(rectangles, copy_img, threshold[1], origin_w, origin_h)

        if len(rectangles) == 0:
            return rectangles

        # ONet处理
        rectangles = self._process_onet_output(rectangles, copy_img, threshold[2], origin_w, origin_h)

        return rectangles

    def _process_pnet_output(self, out, scales, origin_w, origin_h, threshold):
        rectangles = []
        image_num = len(scales)
        for i in range(image_num):
            cls_prob = out[i][0][0][:, :, 1]
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)

            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold)
            rectangles.extend(rectangle)

        return utils.NMS(rectangles, 0.7)

    def _process_rnet_output(self, rectangles, copy_img, threshold, origin_w, origin_h):
        predict_24_batch = [cv2.resize(copy_img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])], (24, 24))
                            for rect in rectangles]
        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)
        cls_prob, roi_prob = out[0], out[1]
        return utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold)

    def _process_onet_output(self, rectangles, copy_img, threshold, origin_w, origin_h):
        predict_batch = [cv2.resize(copy_img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])], (48, 48))
                         for rect in rectangles]
        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob, roi_prob, pts_prob = output[0], output[1], output[2]
        return utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold)
