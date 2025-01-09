import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo


class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        初始化YOLO预测器
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值，低于此值的框将被过滤掉
            nms_threshold: 非极大值抑制（NMS）阈值，用于去除重复的框
            classes_file: 类别名称文件路径
            anchors_file: 先验框（anchors）文件路径
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        # 预读取类别和先验框
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()  # 读取类别名称
        self.anchors = self._get_anchors()  # 读取先验框

        # 为每个类别生成不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)  # 打乱颜色顺序
        random.seed(None)

    def _get_class(self):
        """
        读取类别名称
        返回一个类别名称的列表
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]  # 去掉每行末尾的空格和换行符
        return class_names

    def _get_anchors(self):
        """
        读取先验框（anchors）数据
        返回一个N x 2 的numpy数组，表示每个先验框的宽度和高度
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)  # 将数据按每两个元素分组，表示一个框的宽高
        return anchors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        解码YOLO的输出，计算边框位置和分数
        将YOLO的输出特征图转换为物体框位置及分数
        Parameters
        ----------
            feats: YOLO输出的特征图
            anchors: 先验框
            classes_num: 类别数目
            input_shape: 输入的大小（通常为416x416）
            image_shape: 原始图像大小
        Returns
        -------
            boxes: 物体框的位置（四个坐标）
            boxes_scores: 物体框的分数（置信度 * 类别概率）
        """
        # 获得解码后的box坐标和各个框的分类概率
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 将框转换为原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])  # 将box坐标压缩成一个一维数组，便于后续处理
        # 计算框的分数（置信度 * 类别概率）
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])  # 展平score数组
        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入图片的大小
            image_shape: 原始图片的大小
        Returns
        -------
            boxes: 物体框的坐标，按原图大小调整
        """
        box_yx = box_xy[..., ::-1]  # 反转xy坐标顺序（从(x, y)转为(y, x)）
        box_hw = box_wh[..., ::-1]  # 反转宽高顺序（从(width, height)转为(height, width)）
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        # 计算图片的缩放比例
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        # 计算图像的偏移量和缩放比例
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale  # 调整宽高比例

        # 计算物体框的四个角点坐标
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],  # 左上角x
            box_mins[..., 1:2],  # 左上角y
            box_maxes[..., 0:1],  # 右下角x
            box_maxes[..., 1:2]  # 右下角y
        ], axis=-1)

        # 将框坐标缩放到原始图片的大小
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        解码YOLO最后一层输出的特征图，得到bounding box坐标
        根据YOLO的输出预测框的位置、置信度和类别
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]  # 获取特征图的尺寸
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 创建网格，用来预测框的位置
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)  # 合并网格坐标
        grid = tf.cast(grid, tf.float32)

        # 解码边框的坐标：中心点xy和宽高
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])  # 置信度
        box_class_probs = tf.sigmoid(predictions[..., 5:])  # 类别概率
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
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
        input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        #---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        #---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis = 0)
        box_scores = tf.concat(box_scores, axis = 0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        #---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        #---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)
        return boxes_, scores_, classes_



    #---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    #---------------------------------------#
    def predict(self, inputs, image_shape):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片
            image_shape: 图像原始大小
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train = False)
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training = False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes = 20)
        return boxes, scores, classes