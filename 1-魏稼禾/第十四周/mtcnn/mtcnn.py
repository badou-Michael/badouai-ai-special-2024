from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
import numpy as np
import utils
import cv2

# PNet 检测是否获得人脸，以及bbox
def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])    # 每个样本的形状
    
    # 12,12,3 -> 10,10,10   实际输入是不固定的
    x = Conv2D(10,(3,3), strides=1, padding="valid", name="conv1")(input)
    x = PReLU(shared_axes=[1,2], name="PReLU1")(x)
    # 10,10,10->5,5,10
    x = MaxPool2D(pool_size=2)(x)   # 步幅默认为池化窗口大小
    
    # 5,5,10->3,3,16
    x = Conv2D(16,(3,3),strides=1,padding="valid",name="conv2")(x)
    x = PReLU(shared_axes=[1,2],name="PReLU2")(x)
    
    # 3,3,16->1,1,32
    x = Conv2D(32,(3,3),strides=1,padding="valid",name="conv3")(x)
    x = PReLU(shared_axes=[1,2],name="PReLU3")(x)
    
    # 1,1,2
    classifier = Conv2D(2,(1,1),activation="softmax", name="conv4-1")(x)
    # 1,1,4
    bbox_regress = Conv2D(4,(1,1),name="conv4-2")(x)
    
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# RNet 检测人脸，获得精度更高的框 
def create_Rnet(weight_path):
    input = Input(shape=[24,24,3])
    # 24,24,3->22,22,28->11,11,28
    x = Conv2D(28,(3,3),strides=1, padding="valid", name="conv1")(input)
    x = PReLU(shared_axes=[1,2], name="prelu1")(x)
    x = MaxPool2D(pool_size=(3,3),strides=2,padding="same")(x)
    
    # 11,11,28->9,9,48->4,4,48
    x = Conv2D(48,(3,3),strides=1, padding="valid", name="conv2")(x)
    x = PReLU(shared_axes=[1,2],name="prelu2")(x)
    x = MaxPool2D(pool_size=(3,3), strides=2)(x)
    
    # 4,4,48->3,3,64
    x = Conv2D(64,(2,2),strides=1,padding="valid", name="conv3")(x)
    x = PReLU(shared_axes=[1,2], name="prelu3")(x)
    
    # 3,3,64->64,3,3
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    # 64*3*3->128
    x = Dense(128,name="conv4")(x)
    x = PReLU(name="prelu4")(x)
    classifier = Dense(2,activation="softmax", name="conv5-1")(x)
    bbox_regress = Dense(4,name="conv5-2")(x)
    
    model = Model([input],[classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# ONet
def create_Onet(weight_path):
    input = Input(shape=[48,48,3])
    # 48,48,3->46,46,32->23,23,32
    x = Conv2D(32,(3,3),strides=1,padding="valid",name="conv1")(input)
    x = PReLU(shared_axes=[1,2],name="prelu1")(x)
    x = MaxPool2D(pool_size=3,strides=2,padding="same")(x)
    
    # 23,23,32->21,21,64->10,10,64
    x = Conv2D(64,(3,3),strides=1,padding="valid",name="conv2")(x)
    x = PReLU(shared_axes=[1,2],name="prelu2")(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)
    
    # 10,10,64->8,8,64->4,4,64
    x = Conv2D(64,(3,3),strides=1,padding="valid",name="conv3")(x)
    x = PReLU(shared_axes=[1,2],name="prelu3")(x)
    x = MaxPool2D(pool_size=2)(x)
    
    # 4,4,64->3,3,128
    x = Conv2D(128,(2,2),strides=1,padding="valid",name="conv4")(x)
    x = PReLU(shared_axes=[1,2],name="prelu4")(x)
    
    # 128,3,3
    x = Permute((3,2,1))(x)
    x = Flatten()(x)    # ->1152
    x = Dense(256, name="conv5")(x) # 256
    x = PReLU(name="prelu5")(x)
    
    classifier = Dense(2,activation="softmax",name="conv6-1")(x)
    bbox_regress = Dense(4,name="conv6-2")(x)
    # 脸部的五个特征的坐标
    landmark_regress = Dense(10,name="conv6-3")(x)
    
    model=Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet("model_data/pnet.h5")
        self.Rnet = create_Rnet("model_data/rnet.h5")
        self.Onet = create_Onet("model_data/onet.h5")
        
    def detectFace(self, img, threshold):
        copy_img = (img.copy() - 127.5)/127.5   # 0~255->0~1
        origin_h, origin_w, _ = copy_img.shape
        # 计算原始输入图像每次的缩放比例
        # (0.709,0.709**2,0.709**3,...)
        scales = utils.calculateScales(img)
        out = []
        
        #pnet部分 粗略计算人脸框
        # 在图像金字塔的每一个缩放比例上做预测
        for scale in scales:
            hs = int(origin_h*scale)
            ws = int(origin_w*scale)
            scale_img = cv2.resize(copy_img, (ws,hs))
            inputs = scale_img.reshape(1,*scale_img.shape)
            output = self.Pnet.predict(inputs)  # [分类：bs,h_,w_,2；回归：bs,h_,w_,4]
            out.append(output)
            
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:,:,1]  # 人脸的概率
            roi = out[i][1][0]  # 框的位置
            
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1/scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)
            
        # 继续极大值抑制
        rectangles = utils.NMS(rectangles, 0.7)
        
        if len(rectangles) == 0:
            return rectangles
        
        # rnet部分， 稍微精确计算人脸框
        predict_24_batch = []
        for rectangle in rectangles:
            # 注意x,y和w,h的转换
            # 从初筛的图像中选择检测框
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(24,24))
            predict_24_batch.append(scale_img)
            
        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)
        
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        
        if len(rectangles)==0:
            return rectangles
        
        # 计算人脸框
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48,48))
            predict_batch.append(scale_img)
            
        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]    # 人脸各部分的概率
        
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        
        return rectangles