import colorsys
import os
import random

import numpy as np

import config
from 成元林.第十四周.yolo3.model.yolo3_model import yolo
import tensorflow as tf


class yolo_predict:
    def __init__(self,obj_threshold,nms_threshold,classes_file,anchors_file):
        """
        yolo预测初始化
        @param obj_threshold: 属于某个物体类别的概率阈值
        @param nms_threshold: nms阈值
        @param classes_file: 数据集类别文件 coco数据集有80类
        @param anchors_file: 先验框尺寸读取文件
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.class_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # 画框框用
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #打乱
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        """
        根据路劲读取类别名称
        @return:
        """
        # os.path.expanduser 用于将路径字符串中的波浪线（~）扩展为用户的主目录
        # 它的作用是提供跨平台的路径展开功能，使得路径可以在不同的操作系统上正确地解析
        classes_path = os.path.expanduser(self.class_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        #str.strip()去除前后空格
        class_names = [clsname.strip() for clsname in class_names]
        return class_names

    def _get_anchors(self):
        """
        读取anchors尺寸数据
        @return:
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(",")]
            anchors = np.array(anchors).reshape(-1,2)
        return anchors

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        # inputshape是416x416
        # image_shape是实际图片的大小
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获得置信度box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 其实是解码的过程
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化，相对网格的位置
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    # 获得在原图上框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def predict(self,inputs,image_shape):
        """
        构建预测模型
        @param inputs:处理之后的输入图片
        @param image_shape:图像原始大小
        @return: boxes,scores,classes分别为框坐标，概率值，物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.class_path, pre_train=False)
        output = model.yolo_inference(inputs,config.num_anchors // 3,config.num_classes,training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes