import os
import colorsys
import random
import numpy as np
import config
from model.yolo3_model import yolo
import tensorflow as tf

class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_files):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值
            nms_threshold: nms阈值
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        # 读取种类名称
        self.classes_path = classes_file
        self.class_names = self._get_class()
        # 读取先验框
        self.anchors_path = anchors_files
        self.anchors = self._get_anchors()

        # 画框用
        # 创建了一个HSV(色相、饱和度、亮度)颜色元组的列表, 色相设置为0～1的浮点数
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # 注: map() 函数用于将一个函数应用到可迭代对象
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)
    
    def _get_class(self):
        """
        Introduction
        ------------
            读取类别名称
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()  # 就一行
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2) # 每两个一组, 进行重组
        return anchors

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标, [batch_size, grid_h, grid_w, num_anchors, 2]
            box_wh: 物体框的宽高, 同上
            input_shape: 输入的大小
            image_shape: 图像原始大小
        Returns
        -------
            boxes: 物体框的坐标
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype = tf.float32) # 416,416
        image_shape = tf.cast(image_shape, dtype = tf.float32) # 实际图片大小
        # 按最小比例缩放, 保持长宽比不变, new_shape 表示等比例缩放后图片的新尺寸
        # 缩放后，图片通常不能完全填满 input_shape，所以需要对图片周围进行填充
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
        # 每一侧的填充量,并归一化
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale 

        box_mins = box_yx - (box_hw / 2.) # 左上角
        box_maxes = box_yx + (box_hw / 2.) # 右下角
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2],
        ], axis = -1) # (batch_size, grid_h, grid_w, num_anchors, 4)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo最后一层的输出
            anchors: anchors数量
            num_classes: 类别数量
            input_shape: 输入图片大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs: 物体框坐标, 物体框大小, 物体框概率, 物体类别概率
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1,1,1,num_anchors, 2]) # (1, 1, 1, 3, 2)
        grid_size = tf.shape(feats)[1:3] # [height, width]
        # [batch, height, width, num_anchors * (num_classes + 5)] -> [batch, height, width, num_anchors, num_classes + 5]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 构建13x13x1x2的tensor, 对应每个格子加上对应的坐标, 为了跟prediction做运算
        # tile前 [[[[0]]]],[[[[1]]]],[[[[2]]]] grid_size[0]x1x1x1
        # tile后 (grid_size[0], grid_size, 1, 1)
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        # tile前 [[[[0]]],[[[1]]],[[[2]]],...] 1x13x1x1
        # tile后 13x13x1x1
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1,-1,1,1]), [grid_size[0],1,1,1])
        # 最后一维表示 (x,y坐标，即 weight,height)
        grid = tf.concat([grid_x, grid_y], axis = -1) # (grid_size[0], grid_size, 1, 2) 
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化为 [0, 1]，相对于特征图的整体尺寸
        # predictions[...,:2]: 表示每个网格单元内的anchor的中心点相对于网格左上角的水平、垂直方向偏移量, 加上grid (x,y)就是输入特征图网格中的实际中心点坐标
        # grid_size[::-1]表示 [height, width] -> [width, height]
        # [batch_size, grid_h, grid_w, num_anchors, 2] + [grid_h, grid_w, 1, 2]
        box_xy = (tf.sigmoid(predictions[...,:2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h归一化, [batch, height, width, num_anchors, 2]
        box_wh = tf.exp(predictions[...,2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[...,4:5])
        box_class_probs = tf.sigmoid(predictions[...,5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标,然后计算每个box的分数
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
            boxes_scores: 物体框的分数, 为置信度和类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1,4])
        # 获得置信度 box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """
        Introduction
        ------------
            根据yolo模型的输出进行非极大值抑制,获取最后的物体检测框和物理检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图像原始大小 
            max_boxes: 最大框数量
        Returns
        -------
            boxes_: 物体框坐标
            scores_: 物体类别概率值
            classes_: 物体类别
        """
        # 每个特征层对应3个先验框, 通过 gen_anchors.py kmeans 计算得到
        anchor_mask = [[6, 7, 8],[3, 4, 5],[0, 1, 2]]
        boxes = []
        box_scores = []
        # input_shpae是416x416, 
        input_shape = tf.shape(yolo_outputs[0])[1:3]*32 # 13x32=416
        # yolo_outputs 是一个list,长度为3, 表示3个特征层
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i],
                    self.anchors[anchor_mask[i]], len(self.class_names),
                    input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行里便于操作
        boxes = tf.concat(boxes, axis = 0) # [n, 4]
        # 注: concat之前是一个tensor list,现在要变成一个大的tensor
        box_scores = tf.concat(box_scores, axis = 0) # [n, 80]
        # mask 布尔张量,形状与box_scores相同
        # 大于等于 obj_threshold 的值对应的 mask 位置标记为 True，否则为 False。
        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:,c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:,c], mask[:,c])
            # 非极大值抑制, 方法内部分数从高到低排序
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)

            # 获取非极大值抑制后的结果
            class_boxes =  tf.gather(class_boxes, nms_index) # [num_selected_boxes, 4]
            class_box_scores = tf.gather(class_box_scores, nms_index) # [num_selected_boxes]
            # 框的类型都是c, [num_selected_boxes] 值都是c
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0) # [N,4]
        scores_ = tf.concat(scores_, axis = 0)   # [N]
        classes_ = tf.concat(classes_, axis = 0) # [N]
        return boxes_, scores_, classes_

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
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        # yolo_inference用于获得网络的预测结果
        # 3个特征层, 3个先验框, 80个类别 (即 3x(80+5) = 255)
        output = model.yolo_inference(inputs, config.num_anchors//3, config.num_classes, training = False)
        #  对三个特征层解码, 进行排序并进行非极大抑制
        boxes, scores, classes = self.eval(output, image_shape, max_boxes = 20)
        return boxes, scores, classes