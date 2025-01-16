
from keras import Input, Model
from keras.layers import Conv2D, MaxPool2D, PReLU, Permute, Dense, Flatten
import utils as utils
import cv2
import numpy as np

# ************************* #
# P-Net
# 粗略获取人脸框
# 输出人脸框位置和是否有人脸
# ************************* #
def create_PNet(weight_path):
    input = Input([None, None, 3])

    x = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid", name="conv1")(input)
    x = PReLU(shared_axes=[1, 2], name="PReLU1")(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(16, (3, 3), strides=(1, 1), padding="valid", name="conv2")(x)
    x = PReLU(shared_axes=[1, 2], name="PReLU2")(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding="valid", name="conv3")(x)
    x = PReLU(shared_axes=[1, 2], name="PReLU3")(x)

    classifier = Conv2D(2, (1, 1), activation="softmax", name="conv4-1")(x)
    bbox_regress = Conv2D(4, (1, 1), name="conv4-2")(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# ************************* #
# R-Net
# 非极大值抑制修剪重叠人脸框
# 输出人脸框位置和是否有人脸
# ************************* #
def create_RNet(weight_path):
    # 后面又全连接层，输入固定
    input = Input([24, 24, 3])
    # 24 x 24 x 3 -> 22 x 22 x 28
    x = Conv2D(28, (3, 3), strides=(1, 1), padding="valid", name="conv1")(input)
    x = PReLU(shared_axes=[1, 2], name="prelu1")(x)
    # 22 x 22 x 28 -> 11 x 11 x 28
    x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    # 11 x 11 x 28 -> 8 x 8 x 48
    x = Conv2D(48, (3, 3), strides=(1, 1), padding="valid", name="conv2")(x)
    x = PReLU(shared_axes=[1, 2], name="prelu2")(x)
    # 8 x 8 x 48 -> 4 x 4 x 48
    # x = MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4 x 4 x 48 -> 3 x 3 x 64
    x = Conv2D(64, (2, 2), padding="valid", name="conv3")(x)
    x = PReLU(shared_axes=[1, 2], name="prelu3")(x)

    # 3 x 3 x 64 -> 64 x 3 x 3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name="conv4")(x)
    x = PReLU(name="prelu4")(x)

    # 128 -> 2
    classifier = Dense(2, activation="softmax", name="conv5-1")(x)
    # 128 -> 4
    bbox_regress = Dense(4, activation="softmax", name="conv5-2")(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# ************************* #
# O-Net
# 非极大值抑制修剪重叠人脸框
# 输出人脸框位置和是否有人脸再加上人脸关键点
# ************************* #
def create_ONet(weight_path):
    input = Input([48, 48, 3])

    # 48 x 48 x 3 -> 46 x 46 x 32
    x = Conv2D(32, (3, 3), strides=(1, 1), padding="valid", name="conv1")(input)
    x = PReLU(shared_axes=[1, 2], name="prelu1")(x)
    # 46 x 46 x 32 -> 23 x 23 x 32
    x = MaxPool2D(3, strides=2, padding="same")(x)

    # 23 x 23 x 32 -> 21 x 21 x 64
    x = Conv2D(64, (3, 3), strides=1, padding="valid", name="conv2")(x)
    x = PReLU(shared_axes=[1, 2], name="prelu2")(x)
    # 21 x 21 x 64 -> 10 x 10 x 64
    x = MaxPool2D(3, strides=2, padding="valid")(x)

    # 10 x 10 x 64 -> 4 x 4 x 64
    x = Conv2D(64, (3, 3), strides=1, padding="valid", name="conv3")(x)
    x = PReLU(shared_axes=[1, 2], name="prelu3")(x)
    x = MaxPool2D(2, strides=2, padding="valid")(x)

    # 4 x 4 x 64 -> 3 x 3 x 128
    x = Conv2D(128, (2, 2), strides=1, padding="valid", name="conv4")(x)
    x = PReLU(shared_axes=[1, 2], name="prelu4")(x)

    # 3 x 3 x 128 -> 128 x 3 x 3
    x = Permute([3, 2, 1])(x)
    # 128 x 3 x 3 -> 256
    x = Flatten()(x)
    x = Dense(256, name="conv5")(x)
    x = PReLU(name="prelu5")(x)

    # 256 -> 2
    classifier = Dense(2, activation="softmax", name="conv6-1")(x)
    bbox_regress = Dense(4, name="conv6-2")(x)
    landmark_regress = Dense(10, name="conv6-3")(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path)
    return model

class mtcnn():
    def __init__(self):
        self.PNet = create_PNet('../参考/mtcnn-keras-master/model_data/pnet.h5')
        self.RNet = create_RNet('../参考/mtcnn-keras-master/model_data/rnet.h5')
        self.ONet = create_ONet('../参考/mtcnn-keras-master/model_data/onet.h5')

    def detect_face(self, img, threshold):
        # 将[0, 255]映射到[-1, 1]
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        scales = utils.calculate_scales(copy_img)

        out = []
        # ************************** #
        # PNet 粗略计算人脸框
        # ************************** #
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.PNet.predict(inputs)
            out.append(output)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:,:,1]
            # 对应框位置
            roi = out[i][1][0]

            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)

            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # R-Net部分
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.RNet.predict(predict_24_batch)

        cls_prob = np.array(out[0])
        roi_prob = np.array(out[1])
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if (len(rectangles) == 0):
            return rectangles

        # O-Net部分
        predict_48_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_48_batch.append(scale_img)

        predict_48_batch = np.array(predict_48_batch)
        output = self.ONet.predict(predict_48_batch)

        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles





