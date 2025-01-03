from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import numpy as np
import utils
import cv2

# 构建P-Net:快速生成候选的人脸框。它是一个较小的网络，用于在图像中粗略地检测人脸, 输入的图像尺寸可任意
def create_Pnet(weight_path):
    # 定义模型的输入层，接受任意大小的三通道图像作为输入
    input = Input(shape=[None, None, 3])
    # 定义卷积层、激活函数和池化层
    # N,N,3 -> 5,5,10
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    # 5,5,10 -> 3,3,16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    # 3,3,16 -> 1,1,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 第一部分输出， 用于判断该图是否存在人脸(1,1,2)
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 第二部分输出，框的精确位置（x,y,w,h）(1,1,4)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    # 加载预训练权重 by_name=True：这个参数告诉Keras在加载权重时，根据层的名称来匹配权重，而不是根据层的顺序
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 构建R-Net:对P-Net生成的人脸框进行进一步的筛选和精确定位, 输入的图像尺寸固定
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

    # 3,3,64 -> 64,3,3；Permute层用于重新排列输入张量的维度；Flatten层用于将多维输入张量展平成一维张量
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)
    # 128 -> 2 128 -> 4
    # 第一部分输出， 用于判断该图是否存在人脸(2)
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 第二部分输出，框的精确位置（x,y,w,h）(4)
    bbox_regress = Dense(4, name='conv5-2')(x)
    # 加载预训练权重
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# 构建O-Net: 用于精确地定位人脸和关键点, 输入的图像尺寸固定
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
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2 第一部分输出， 用于判断该图是否存在人脸(2)
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    # 256 -> 4 第二部分输出，框的精确位置（x,y,w,h）(4)
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 256 -> 10 第三部分输出，人脸五个关键点位置（x,y）*5 ->(10)
    landmark_regress = Dense(10, name='conv6-3')(x)
    # 加载预训练权重
    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model

# 定义mtcnn模型的类：将三个网络整合到一个MTCNN类中，实现人脸检测的流程
class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')


    def detectFace(self, img, threshold):
        # 图片归一化，加速收敛
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape#获取原始图像尺寸
        # 记录图像缩放比例
        scales = utils.calculateScales(img)

        out = []
        for scale in scales:
            # 将原始图像按照缩放比例进行缩放
            scale_img = cv2.resize(copy_img, (int(origin_w * scale), int(origin_h * scale)))
            # 将图像大小进行重塑
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
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
            # 对pnet处理后的结果进行处理 获得人脸和框的位置信息
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        # 进行非极大抑制NMS
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # 对Rnet处理后的结果进行处理， 获得更加精确的人脸和框的信息
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

        # # 对Onet处理后的结果进行处理，获得最终的人脸、框和关键点位置信息
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

