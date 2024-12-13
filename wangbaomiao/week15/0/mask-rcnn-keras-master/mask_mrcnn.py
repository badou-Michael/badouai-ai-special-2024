# -*- coding: utf-8 -*-
# time: 2024/11/29 09:44
# file: mask_mrcnn.py
# author: flame
import os
import keras.backend as K
import numpy as np
from nets.mrcnn import get_predict_model
from utils import visualize
from utils.anchors import get_anchors
from utils.config import Config
from utils.utils import mold_inputs, unmold_detections


class MASK_RCCN(object):
    '''
    MASK_RCNN模型的配置和检测类。
    该类用于加载预训练的MASK_RCNN模型，并进行图像实例分割的任务。
    它包含模型的默认配置参数，模型初始化，图像检测等功能。
    '''

    ''' 默认的模型配置参数，包括模型路径、类别文件路径、置信度阈值等。 
     类别文件路径，默认为 'model_data/coco_classes.txt'。 
     模型权重文件路径，默认为 'model_data/mask_rcnn_coco.h5'。 
     置信度阈值，默认为 0.7，表示检测结果的最小置信度。 
     RPN 锚点尺度，默认为 (32, 64, 128, 256, 512)，表示不同尺度的锚点
     图像最小维度，默认为 1024，表示输入图像的最小边长。 
     图像最大维度，默认为 1024，表示输入图像的最大边长。'''
    _defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',
        "classes_path": 'model_data/coco_classes.txt',
        "confidence": 0.7,
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
        "IMAGE_MIN_DIM": 1024,
        "IMAGE_MAX_DIM": 1024,
    }

    @classmethod
    def get_defaults(cls, n):
        ''' 获取默认配置参数，如果参数不存在则返回未识别的属性名。 '''
        ''' 判断参数 n 是否在默认配置字典中。 '''
        if n in cls._defaults:
            ''' 如果存在，返回对应的默认值。 '''
            return cls._defaults[n]
        else:
            ''' 如果不存在，返回未识别的属性名。 '''
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        ''' 初始化 MASK_RCCN 类，更新默认配置并加载模型。 '''
        ''' 更新默认配置参数。 '''
        self.__dict__.update(self._defaults)
        ''' 加载类别名称列表。 '''
        self.class_names = self._get_class()
        ''' 获取 Keras 会话。 '''
        self.sess = K.get_session()
        ''' 获取模型配置。 '''
        self.config = self._get_config()
        ''' 生成模型。 '''
        self.generate()

    def _get_class(self):
        ''' 从类别文件中读取类别名称列表，并在开头插入背景类别 'BG'。 '''
        ''' 扩展用户路径，确保路径正确。 '''
        classes_path = os.path.expanduser(self.classes_path)
        ''' 打开类别文件并读取所有行。 '''
        with open(classes_path) as f:
            class_names = f.readlines()
        ''' 去除每行末尾的换行符。 '''
        class_names = [c.strip() for c in class_names]
        ''' 在类别名称列表的开头插入背景类别 'BG'。 '''
        class_names.insert(0, 'BG')
        ''' 返回类别名称列表。 '''
        return class_names

    def _get_config(self):
        ''' 创建并返回模型的配置对象。 '''
        ''' 定义一个内部类 InferenceConfig 继承自 Config。 '''
        class InferenceConfig(Config):
            ''' 配置名称，默认为 'shapes'。 '''
            NAME = "shapes"
            ''' 类别数量，等于类别名称列表的长度。 '''
            NUM_CLASSES = len(self.class_names)
            ''' GPU 数量，默认为 1。 '''
            GPU_COUNT = 1
            ''' 每个 GPU 处理的图像数量，默认为 1。 '''
            IMAGES_PER_GPU = 1
            ''' 检测结果的最小置信度，默认为 self.confidence。 '''
            DETECTION_MIN_CONFIDENCE = self.confidence
            ''' RPN 锚点尺度，默认为 self.RPN_ANCHOR_SCALES。 '''
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            ''' 图像最小维度，默认为 self.IMAGE_MIN_DIM。 '''
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            ''' 图像最大维度，默认为 self.IMAGE_MAX_DIM。 '''
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM
        ''' 创建 InferenceConfig 实例。 '''
        config = InferenceConfig()
        ''' 显示配置信息。 '''
        config.display()
        ''' 返回配置对象。 '''
        return config

    def generate(self):
        ''' 生成并加载模型。 '''
        ''' 扩展用户路径，确保路径正确。 '''
        model_path = os.path.expanduser(self.model_path)
        ''' 断言模型路径以 .h5 结尾，确保模型文件格式正确。 '''
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        ''' 获取类别数量。 '''
        self.num_classes = len(self.class_names)
        ''' 获取预测模型。 '''
        self.model = get_predict_model(self.config)
        ''' 加载模型权重。 '''
        self.model.load_weights(self.model_path, by_name=True)

    def detect_image(self, image):
        ''' 对输入图像进行实例分割检测。 '''
        ''' 将输入图像转换为 numpy 数组。 '''
        image = [np.array(image)]
        ''' 将输入图像调整为模型输入格式。 '''
        molded_images, image_metas, windows = mold_inputs(self.config, image)
        ''' 获取调整后的图像形状。 '''
        image_shape = molded_images[0].shape
        ''' 生成锚点。 '''
        anchors = get_anchors(self.config, image_shape)
        ''' 广播锚点到所需形状。 '''
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
        ''' 使用模型进行预测，返回多个输出，其中只有 `detections` 和 `mrcnn_mask` 是需要的。 '''
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        ''' 解析预测结果，将模型输出转换为实际的检测框、类别 ID、置信度分数和掩码。 '''
        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0],
                              mrcnn_mask[0], image[0].shape, molded_images[0].shape, windows[0])

        ''' 构建检测结果字典，方便后续处理和可视化。
         检测框的坐标。检测到的类别的 ID。检测框的置信度分数。检测框的掩码。'''
        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        ''' 可视化检测结果，显示图像上的检测框、类别标签和掩码。 '''
        visualize.display_instances(image[0], r['rois'],r['masks'], r['class_ids'], self.class_names, r['scores'])

    def close_session(self):
        ''' 关闭 Keras 会话。 '''
        self.sess.close()
