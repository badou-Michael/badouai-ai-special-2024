from keras.layers import Conv2D,Input,MaxPooling2D,Permute,Flatten,Dense
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import cv2
import utils
import numpy as np


def create_pnet(weight_path):
    input = Input(shape=[None,None,3])

    x = Conv2D(filters=10,kernel_size=(3,3),strides=(1,1),padding="valid",name="p_conv1")(input)
    x = PReLU(shared_axes=[1,2],name="p_prelu1")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding="valid",name="p_conv2")(x)
    x = PReLU(shared_axes=[1,2],name="p_prelu2")(x)

    x = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding="valid",name = "p_conv3")(x)
    x = PReLU(shared_axes=[1, 2], name="p_prelu3")(x)

    #输出人脸特征，进行二分类
    face_class = Conv2D(filters=2,kernel_size=(1,1),strides=(1,1),activation='softmax',name = "p_conv4")(x)

    # 无激活函数，线性。
    bbox_regress = Conv2D(4, (1, 1), name='conv4-1')(x)
    model = Model([input],[face_class,bbox_regress])
    model.load_weights(weight_path)
    return model

def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])

    x = Conv2D(filters=28, kernel_size=(3, 3), strides=(1, 1), padding="valid", name="r_conv1")(input)
    x = PReLU(shared_axes=[1,2],name="r_prelu1")(x)
    x = MaxPooling2D(pool_size=3,strides=2, padding='same')(x)

    x = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="valid", name="r_conv2")(x)
    x = PReLU(shared_axes=[1, 2], name="r_prelu2")(x)
    x = MaxPooling2D(pool_size=3,strides=2)(x)

    x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding="valid", name="r_conv3")(x)
    x = PReLU(shared_axes=[1, 2], name="r_prelu3")(x)

    #3x3x64 => 64x3x3
    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    x = Dense(128, name='r-full-connect')(x)
    x = PReLU(name='r_prelu4')(x)

    face_class = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [face_class, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Onet(weight_path):
    input = Input(shape=[48, 48, 3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPooling2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model

class Mtcnn():
    def __init__(self):
        self.pnet = create_pnet("model_data/pnet.h5")
        self.rnet = create_Rnet("model_data/rnet.h5")
        self.onet = create_Onet("model_data/onet.h5")

    def detectFace(self, img, threshold):
        #-----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        #-----------------------------#

        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        #-----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        #-----------------------------#
        scales = utils.calculateScales(img)

        out = []
        #-----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        #-----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.pnet.predict(inputs)
            # 将所有output加入out
            out.append(output)

        image_num = len(scales)
        rectangles = []
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

        #-----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        #-----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        #-----------------------------#
        #   计算人脸框
        #   onet部分
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles


