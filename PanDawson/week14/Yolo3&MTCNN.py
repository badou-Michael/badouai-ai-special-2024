def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否为训练模式
        """
        conv_index = 1
        # Conv2D Block5L 1024
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            #--------------------------------------#
            #   获得第一个特征层
            #--------------------------------------#
            # 这里对应PPT中最底下那个特征层（从下往上，第一第二第三个特征层）
            # PPT中使用的是voc数据集，类20.这里用的coco数据集，类80
            # conv2D 3*3 + Conv2D 1*1
            # conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   获得第二个特征层
            #--------------------------------------#
            # 需要先上采样到大小一样才能做concat 26，26，256
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # unSample_0 = 26,26,256
            # 上采样方式：最邻近插值
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            # route0 = 26,26,768
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = 'route_0')
            # conv2d_65 = 52,52,256，conv2d_67 = 26,26,255(3x(80+5))
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   获得第三个特征层
            #--------------------------------------# 
            # 需要先上采样到大小一样才能做concat 52，52，128
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # unSample_1 = 52,52,128
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            # route1= 52,52,384
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            # conv2d_75 = 52,52,255
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]


# Found 29 boxes for img
# car 0.53 (206, 3) (242, 73)
# (72, 15)
# car 0.66 (572, 0) (628, 69)
# (72, 15)
# car 0.74 (317, 0) (433, 52)
# (72, 15)
# car 0.82 (229, 2) (281, 58)
# (72, 15)
# car 0.97 (20, 117) (324, 230)
# (72, 16)
# car 0.98 (469, 5) (602, 84)
# (72, 16)
# car 0.98 (75, 9) (221, 118)
# (72, 16)
# car 0.99 (279, 11) (410, 128)
# (72, 16)
# car 0.99 (346, 44) (626, 179)
# (72, 16)
# car 0.99 (498, 236) (676, 393)
# (72, 16)
# person 0.59 (335, 186) (357, 247)
# (99, 18)
# person 0.59 (1, 210) (29, 358)
# (99, 18)
# person 0.68 (339, 202) (382, 319)
# (99, 18)
# person 0.77 (454, 195) (504, 330)
# (99, 18)
# person 0.78 (727, 204) (782, 370)
# (99, 18)
# person 0.85 (633, 190) (679, 283)
# (99, 18)
# person 0.85 (111, 218) (170, 356)
# (99, 18)
# person 0.87 (368, 201) (417, 299)
# (99, 18)
# person 0.87 (706, 199) (753, 367)
# (99, 18)
# person 0.91 (560, 190) (617, 252)
# (99, 18)
# person 0.91 (64, 194) (115, 320)
# (99, 18)
# person 0.94 (170, 192) (217, 342)
# (99, 18)
# person 0.95 (416, 192) (465, 343)
# (99, 18)
# person 0.96 (752, 208) (798, 377)
# (99, 18)
# person 0.97 (139, 205) (190, 353)
# (99, 18)
# person 0.97 (671, 209) (725, 367)
# (99, 18)
# person 0.97 (293, 202) (348, 353)
# (99, 18)
# person 0.97 (222, 212) (292, 360)
# (99, 18)
# person 0.99 (19, 225) (79, 374)
# (99, 18)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=800x485 at 0x7F5F96FDF520>


from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2
#-----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
#-----------------------------#
def create_Pnet(weight_path):
    #12*12*3
    input = Input(shape=[None, None, 3])
    #5*5*10
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # 小于0时 为ax,而不是0
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    #3*3*16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)
    #1*1*32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    # 1*1*2
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    # 1*1*4
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

#-----------------------------#
#   mtcnn的第二段
#   精修框
#-----------------------------#
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

#-----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
#-----------------------------#
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
        #用于生成候选框
        self.Pnet = create_Pnet('model_data/pnet.h5')
        #用于筛选候选框
        self.Rnet = create_Rnet('model_data/rnet.h5')
        #用于最终的人脸检测和关键点定位
        self.Onet = create_Onet('model_data/onet.h5')

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
            # 图像金字塔只用在推理过程，反向传播使用
            output = self.Pnet.predict(inputs)
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
        out = self.Rnet.predict(predict_24_batch)

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
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles

# (245, 440)
# (172, 310)
# (120, 218)
# (84, 153)
# (58, 107)
# (39, 74)
# (26, 51)
# (17, 35)
# (10, 23)
# (6, 15)
# (3, 9)
