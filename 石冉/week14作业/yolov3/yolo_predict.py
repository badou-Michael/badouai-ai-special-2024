import os
import tensorflow as tf
import week14_yolo_config
import random
import colorsys
import numpy as np
from week14_yolo3_model import yolo

class yolo_predictor:
    def __init__(self,obj_threshold,nms_threshold,classes_file,anchors_file):
        '''obj_threshold:目标检测为物体的阈值
        nms_threshold:nms阈值'''
        self.obj_threshold=obj_threshold
        self.nms_threshold=nms_threshold
        #预读取
        self.classes_path=classes_file
        self.anchors_path=anchors_file
        #读取种类名称
        self.class_names=self._get_class()
        self.anchors=self._get_anchors()

        #画框用
        #建了一个HSV颜色元组的列表。对于self.class_names中的每个类名，它生成一个HSV元组，
        # 其中色相（H）值是x / len(self.class_names)，饱和度（S）和明度（V）都被设置为1，意味着完全饱和和完全不透明。
        hsv_tuples=[(x/len(self.class_names),1.,1.) for x in range(len(self.class_names))]
        #将hsv_tuples中的每个HSV元组转换为RGB元组
        self.colors=list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
        #将上一步得到的RGB元组中的每个浮点数值（范围在0到1之间）转换为0到255之间的整数
        self.colors=list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255))))
        #通过设置随机种子，代码确保了颜色的生成是可重复的，这对于调试和测试可能是有用的。
        # 之后，通过打乱颜色列表，代码确保了颜色的分配是随机的，这有助于在可视化或绘图时避免任何潜在的偏见。
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        '''读取类别名称'''
        #os.path.expanduser 函数会将路径中的 ~ 替换为当前用户的主目录路径，这样即使在不同用户的电脑上，路径也能正确解析
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''读取anchors数据'''
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readlines()
            anchors=[float(x) for x in anchors.split(',')]
            # reshape(-1, 2) 方法将数组重塑为一个二维数组，其中 -1 表示自动计算行数，以确保每行有两个元素。
            # 这通常对应于锚点的宽度和高度。
            anchors=np.array(anchors).reshape(-1,2)
        return anchors

    #对三个特征层进行解码，进行排序并进行非极大值抑制
    def boxes_and_scores(self,feats,anchors,class_num,input_shape,image_shape):
        '''introduction:将预测出的box坐标转换为原图坐标，然后计算每个box的分数
        feats:yolo输出的feature map
        anchors:anchors的位置
        class_num：类别数目
        input_shape：输入大小
        image_shape：图片大小
        return:boxes：物体框的位置，boxes_scores：物体框的分数=置信度*类别概率'''

        #获得特征
        box_xy,box_wh,box_confidence,box_class_probs=self._get_feats(feats,anchors,class_num,input_shape)
        #寻找原图位置
        boxes=self.correct_boxes(box_xy,box_wh,input_shape,image_shape)
        boxes=tf.reshape(boxes,[-1,4])
        #获得boxes_scores
        boxes_scores=box_confidence*box_class_probs
        boxes_scores=tf.reshape(boxes_scores,[-1,class_num])
        return boxes, boxes_scores

    def correct_boxes(self,box_xy,box_wh,input_shape,image_shape):
        '''introduction：计算物体框预测坐标在原图中的坐标
        parameters:
        box_xy:物体框左上角坐标
        box_wh:物体框的宽高
        input_shape:输入大小
        image_shape:图片大小
        return: boxes:物体框的位置'''
        #将box_xy数组的最后一个维度的元素顺序反转。如果box_xy表示的是边界框的坐标（x, y），那么box_yx将会是（y, x）。
        # 这种转换可能是为了满足某些函数或库的特定输入要求，它们可能需要边界框的坐标以（y, x）的顺序来表示
        box_yx=box_xy[...,::-1]
        box_hw=box_wh[...,::-1]

        #416,416
        input_shape=tf.cast(input_shape,dtype=tf.float32)#转为浮点数
        image_shape=tf.cast(image_shape,dtype=tf.float32)#同上
        #计算调整后的图像尺寸，使得调整后的图像尺寸与输入图像尺寸的最小比例相匹配，并四舍五入到最近的整数
        new_shape=tf.round(image_shape*tf.reduce_min(input_shape/image_shape))
        #计算偏移量和缩放比例。
        offset=(input_shape-new_shape)/2./input_shape
        scale=input_shape/new_shape
        # 应用缩放和平移操作，将物体框的坐标调整到原始输入图像的空间
        box_yx=(box_yx-offset)*scale
        box_hw*=scale
        #计算物体框的最小和最大坐标，即物体框的左上角和右下角坐标
        box_mins=box_yx-(box_hw/2.)
        box_maxes=box_yx+(box_hw/2.)
        #将物体框的最小和最大坐标合并成一个四元组，表示物体框的位置
        boxes=tf.concat([
            box_mins[...,0:1],
            box_mins[...,1:2],
            box_maxes[...,0:1],
            box_maxes[...,1:2]
        ],axis=-1)
        #将物体框的坐标乘以图像的尺寸，将坐标从输入图像空间转换到原始图像空间
        boxes*=tf.concat([image_shape,image_shape],axis=-1)
        return boxes

    def _get_feats(self,feats,anchors,num_classes,input_shape):
        '''introduction:根据yolo最后一层的输出，确定bouding box
        parameter:
        feats:yolo最后一层的输出
        anchors：anchors的位置
        num_classes:类别数量
        input_shape:输入大小
        return：
        box_xy,box_wh,box_confidence,box_class_probs'''

        #获取锚点（anchors）的数量
        num_anchors=len(anchors)
        #将锚点转换为张量，并调整其形状以匹配网络输出的维度
        anchors_tensor=tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,num_anchors,2])
        #获取网络输出的特征图的宽度和高度
        grid_size=tf.shape(feats)[1:3]
        #将网络输出的特征图调整为合适的形状，以便于提取边界框和类别信息
        predictions=tf.reshape(feats,[-1,grid_size[0],grid_size[1],num_anchors,num_classes+5])
        #创建一个与特征图大小相同的网格，用于计算边界框的中心坐标
        grid_y=tf.tile(tf.reshape(tf.range(grid_size[0]),[-1,1,1,1]),[1,grid_size[1],1,1])
        grid_x=tf.tile(tf.reshape(tf.range(grid_size[1]),[1,-1,1,1]),[grid_size[0],1,1,1])
        #将x和y坐标合并为一个网格。
        grid=tf.concat([grid_x,grid_y],axis=-1)
        #将网格的数据类型转换为浮点数
        grid=tf.cast(grid,tf.float32)
        #使用sigmoid函数激活预测的x和y坐标，然后加上网格的坐标，最后除以特征图的宽度和高度，得到边界框中心的归一化坐标
        box_xy=(tf.sigmoid(predictions[...,:2])+grid)/tf.cast(grid_size[::-1],tf.float32)
        #使用指数函数激活预测的宽度和高度，然后乘以锚点的宽度和高度，再除以输入图像的宽度和高度，得到边界框的归一化宽度和高度
        box_wh=tf.exp(predictions[...,2:4])*anchors_tensor/tf.cast(input_shape[::-1],tf.float32)
        #使用sigmoid函数激活预测的置信度
        box_confidence=tf.sigmoid(predictions[...,4:5])
        #使用sigmoid函数激活预测的类别概率
        box_class_probs=tf.sigmoid(predictions[...,5:])
        return box_xy,box_wh,box_confidence,box_class_probs

    def eval(self,yolo_outputs,image_shape,max_boxes=20):
        '''introduction:根据yolo模型的输出进行非极大值抑制，获取最后的物体检测框和检测类别
        patameters：
        yolo_outputs:yolo模型输出
        image_shape:图片大小
        max_boxes:最大box数量
        return：
        boxes:物体框的位置
        scores:物体类别的概率
        classes:物体类别'''

        #每一个特征层对应3个先验框，指定每个特征层使用的先验框的索引
        anchor_mask=[[6,7,8],[3,4,5],[0,1,2]]
        #初始化两个空列表，boxes用于存储所有检测框的坐标，box_scores用于存储所有检测框的得分
        boxes=[]
        box_scores=[]
        #计算YOLO模型的输入尺寸。这里假设模型的输入尺寸是416x416，
        # yolo_outputs[0]是模型的第一个输出层，tf.shape获取其尺寸，然后乘以32（因为YOLO的输入是32倍于原始图像尺寸）
        input_shape=tf.shape(yolo_outputs[0])[1:3]*32

        #遍历YOLO模型的每个输出层，使用self.boxes_and_scores方法来解码每个预测框的坐标和得分，
        # 并将结果添加到boxes和box_scores列表中
        for i in range(len(yolo_outputs)):
            _boxes,_box_scores=self.boxes_and_scores(yolo_outputs[i],self.anchors[anchor_mask[i]],len(self.class_names),input_shape,image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        #使用tf.concat将所有特征层的检测框和得分合并成一个张量，便于后续操作。
        boxes=tf.concat(boxes,axis=0)
        box_scores=tf.concat(box_scores,axis=0)
        #创建一个布尔掩码mask，用于过滤出得分大于对象检测阈值self.obj_threshold的框。
        mask=box_scores>=self.obj_threshold
        #max_boxes_tensor是一个常量张量，表示最大检测框数量
        max_boxes_tensor=tf.constant(max_boxes,dtype=tf.int32)
        #初始化三个空列表，用于存储最终的检测框、得分和类别。
        boxes_=[]
        scores_=[]
        classes_=[]

        #取出每一类阈值大于obj_threshold的框和得分，并对得分进行非极大值抑制
        for c in range(len(self.class_names)):
            #取出所有类为c的box
            class_boxes=tf.boolean_mask(boxes,mask[:,c])
            #取出所有类为c的分数
            class_box_scores=tf.boolean_mask(box_scores[:,c],mask[:,c])
            #非极大值抑制
            nms_index=tf.image.non_max_suppression(class_boxes,class_box_scores,max_boxes_tensor,iou_threshold=self.nms_threshold)
            #获取非极大抑制的结果
            class_boxes=tf.gather(class_boxes,nms_index)
            class_box_scores=tf.gather(class_box_scores,nms_index)
            classes=tf.ones_like(class_box_scores,'int32')*c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.apend(classes)
        #使用tf.concat将所有类别的检测结果合并，并返回这些张量
        boxes_=tf.concat(boxes_,axis=0)
        scores_=tf.concat(scores_,axis=0)
        classes_=tf.concat(classes_,axis=0)
        return boxes_,scores_,classes_

    #predict用于预测，分为三步：建立yolo对象，获得预测结果，对预测结果进行处理
    def predict(self,inputs,image_shape):
        '''introduction:构建预测模型
        parameters：
        inputs:处理之后的输入图片
        image_shape:图像原始大小
        return：
        boxes:物体框坐标
        scores:物体概率值
        classes：物体类别'''
        model=yolo(week14_yolo_config.norm_epsilon,week14_yolo_config.norm_decay,self.anchors_path,self.classes_path,pre_train=False)
        output=model.yolo_inferrence(inputs,week14_yolo_config.num_anchors//3,week14_yolo_config.num_classes,training=False)
        boxes,scores,classes=self.eval(output,image_shape,max_boxes=20)
        return boxes,scores,classes


