# -*- coding: utf-8 -*-
# time: 2024/11/28 15:47
# file: mrcnn.py
# author: flame
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Add, Lambda, Concatenate
from keras.layers import  Reshape, TimeDistributed, Dense, Conv2DTranspose
from keras.models import Model
from nets.resnet import get_resnet
from nets.layers import ProposalLayer, PyramidROIAlign,DetectionLayer
from nets.mrcnn_training import *
from utils.anchors import get_anchors
import keras.backend as K


''' 
区域建议网络(RPN)的图形函数，用于生成候选区域的类别和边界框预测。
输入特征图和每个位置的锚点数量，输出RPN类别逻辑、类别概率和边界框预测。
'''

def rpn_graph(feature_map, anchors_per_location):
    ''' 定义共享卷积层，使用512个3x3卷积核，激活函数为ReLU，用于提取特征图的高级特征。 '''
    shared = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)

    ''' 定义类别预测层，使用2 * anchors_per_location个1x1卷积核，激活函数为线性，用于生成类别预测。 '''
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    ''' 将类别预测结果 reshape 成 (batch_size, -1, 2)，其中-1表示所有锚点的数量。 '''
    rpn_class_logits = Reshape(([-1, 2]), name='rpn_class_logits')(x)

    ''' 使用 softmax 激活函数将类别逻辑转换为类别概率。 '''
    rpn_probs = Activation('softmax', name='rpn_class_xxx')(rpn_class_logits)

    ''' 定义边界框预测层，使用4 * anchors_per_location个1x1卷积核，激活函数为线性，用于生成边界框预测。 '''
    x = Conv2D(4 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_bbox_pred')(shared)

    ''' 将边界框预测结果 reshape 成 (batch_size, -1, 4)，其中-1表示所有锚点的数量。 '''
    rpn_bbox = Reshape(([-1, 4]), name='rpn_bbox')(x)

    ''' 返回 RPN 类别逻辑、类别概率和边界框预测。 '''
    return [rpn_class_logits, rpn_probs, rpn_bbox]

''' 
构建RPN模型，输入特征图和每个位置的锚点数量，输出RPN模型。
'''

def build_rpn_model(anchors_per_location, depth):
    ''' 定义输入特征图，形状为 (batch_size, height, width, depth)。 '''
    input_feature_map = Input(shape=[None, None, depth], name='input_rpn_feature_map')

    ''' 调用 rpn_graph 函数生成 RPN 的输出。 '''
    outputs = rpn_graph(input_feature_map, anchors_per_location)

    ''' 构建并返回 RPN 模型。 '''
    return Model([input_feature_map], outputs, name='rpn_model')

''' 
特征金字塔网络(FPN)分类器图形函数，用于生成 ROI 的类别和边界框预测。
输入 ROIs、特征图、图像元数据、池化大小、类别数量等参数，输出类别逻辑、类别概率和边界框预测。
'''

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    ''' 使用 PyramidROIAlign 层对 ROIs 进行池化操作，生成固定大小的特征图。 '''
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_classifier')([rois, image_meta] + feature_maps)

    ''' 使用 TimeDistributed 应用 3x3 卷积层，提取特征图的高级特征，激活函数为 ReLU。 '''
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding='valid'), name='mrcnn_class_conv1')(x)

    ''' 使用 TimeDistributed 应用批量归一化层，训练模式由 train_bn 控制。 '''
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)

    ''' 使用 ReLU 激活函数激活特征图。 '''
    x = Activation('relu')(x)

    ''' 使用 TimeDistributed 应用 1x1 卷积层，进一步提取特征图的高级特征，激活函数为 ReLU。 '''
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)), name='mrcnn_class_conv2')(x)

    ''' 使用 TimeDistributed 应用批量归一化层，训练模式由 train_bn 控制。 '''
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)

    ''' 使用 ReLU 激活函数激活特征图。 '''
    x = Activation('relu')(x)

    ''' 使用 Lambda 层压缩特征图的维度，从 (batch_size, num_rois, 1, 1, fc_layers_size) 压缩到 (batch_size, num_rois, fc_layers_size)。 '''
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name='pool_squeeze')(x)

    ''' 使用 TimeDistributed 应用全连接层，生成类别逻辑输出。 '''
    mrcnn_class_logits = TimeDistributed(Dense(num_classes), name='mrcnn_class_logits')(shared)

    ''' 使用 TimeDistributed 应用 softmax 激活函数，将类别逻辑转换为类别概率。 '''
    mrcnn_probs = TimeDistributed(Activation("softmax"), name='mrcnn_class')(mrcnn_class_logits)

    ''' 使用 TimeDistributed 应用全连接层，生成边界框预测，激活函数为线性。 '''
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)

    ''' 将边界框预测结果 reshape 成 (batch_size, num_rois, num_classes, 4)。 '''
    mrcnn_bbox = Reshape([-1, num_classes, 4], name='mrcnn_bbox')(x)

    ''' 返回 MRCNN 类别逻辑、类别概率和边界框预测。 '''
    return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

'''
构建FPN掩码图。
该函数通过金字塔ROI对齐和卷积操作来生成掩码。
参数:
- rois: 区域建议框
- feature_maps: 特征图列表
- image_meta: 图像元数据
- pool_size: 池化大小
- num_classes: 类别数量
- train_bn: 是否训练批量归一化层
返回值: 掩码图
'''
def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    ''' 使用 PyramidROIAlign 层从特征图中提取指定区域的特征，并进行池化。 '''
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_mask')([rois, image_meta] + feature_maps)
    ''' 使用 TimeDistributed 包装的 Conv2D 层进行卷积操作，输出通道数为 256。 '''
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'), name='mask_mask_conv1')(x)
    ''' 使用 TimeDistributed 包装的 BatchNormalization 层进行批量归一化，是否训练由 train_bn 参数决定。 '''
    x = TimeDistributed(BatchNormalization(), name='mask_mask_bn1')(x, training=train_bn)
    ''' 使用 ReLU 激活函数激活前一层的输出。 '''
    x = Activation('relu')(x)
    ''' 使用 TimeDistributed 包装的 Conv2D 层进行卷积操作，输出通道数为 256。 '''
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'), name='mask_mask_conv2')(x)
    ''' 使用 TimeDistributed 包装的 BatchNormalization 层进行批量归一化，是否训练由 train_bn 参数决定。 '''
    x = TimeDistributed(BatchNormalization(), name='mask_mask_bn2')(x, training=train_bn)
    ''' 使用 ReLU 激活函数激活前一层的输出。 '''
    x = Activation('relu')(x)
    ''' 使用 TimeDistributed 包装的 Conv2D 层进行卷积操作，输出通道数为 256。 '''
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'), name='mask_mask_conv3')(x)
    ''' 使用 TimeDistributed 包装的 BatchNormalization 层进行批量归一化，是否训练由 train_bn 参数决定。 '''
    x = TimeDistributed(BatchNormalization(), name='mask_mask_bn3')(x, training=train_bn)
    ''' 使用 ReLU 激活函数激活前一层的输出。 '''
    x = Activation('relu')(x)
    ''' 使用 TimeDistributed 包装的 Conv2D 层进行卷积操作，输出通道数为 256。 '''
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'), name='mask_mask_conv4')(x)
    ''' 使用 TimeDistributed 包装的 BatchNormalization 层进行批量归一化，是否训练由 train_bn 参数决定。 '''
    x = TimeDistributed(BatchNormalization(), name='mask_mask_bn4')(x, training=train_bn)
    ''' 使用 ReLU 激活函数激活前一层的输出。 '''
    x = Activation('relu')(x)
    ''' 使用 TimeDistributed 包装的 Conv2DTranspose 层进行反卷积操作，输出通道数为 256，步长为 2，激活函数为 ReLU。 '''
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv')(x)
    ''' 使用 TimeDistributed 包装的 Conv2D 层进行卷积操作，输出通道数为 num_classes，激活函数为 Sigmoid。 '''
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'), name='mrcnn_mask')(x)
    return x

'''
获取预测模型。
根据配置构建 Mask R-CNN 模型用于预测。
参数:
- config: 配置对象，包含模型配置参数
返回值: 预测模型
'''
def get_predict_model(config):
    ''' 获取图像的高度和宽度。 '''
    h, w = config.IMAGE_SHAPE[:2]
    ''' 检查图像尺寸是否可以被 2 至少整除 6 次，否则抛出异常。 '''
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be divisible by 2 at least 6 times")
    ''' 定义输入图像的形状。 '''
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')
    ''' 定义输入图像元数据的形状。 '''
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name='input_image_meta')
    ''' 定义输入锚点的形状。 '''
    input_anchors = Input(shape=[None, 4], name="input_anchors")
    ''' 使用 ResNet 架构获取特征图 C2, C3, C4, C5。 '''
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)
    ''' 使用 Conv2D 层将 C5 转换为 P5。 '''
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    ''' 将 P5 上采样并与 C4 进行加和操作，得到 P4。 '''
    P4 = Add(name='fpn_p4add')([UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    ''' 将 P4 上采样并与 C3 进行加和操作，得到 P3。 '''
    P3 = Add(name='fpn_p3add')([UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    ''' 将 P3 上采样并与 C2 进行加和操作，得到 P2。 '''
    P2 = Add(name='fpn_p2add')([UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

    ''' 使用 Conv2D 层进一步处理 P2，输出通道数为 TOP_DOWN_PYRAMID_SIZE。 '''
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p2')(P2)
    ''' 使用 Conv2D 层进一步处理 P3，输出通道数为 TOP_DOWN_PYRAMID_SIZE。 '''
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p3')(P3)
    ''' 使用 Conv2D 层进一步处理 P4，输出通道数为 TOP_DOWN_PYRAMID_SIZE。 '''
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p4')(P4)
    ''' 使用 Conv2D 层进一步处理 P5，输出通道数为 TOP_DOWN_PYRAMID_SIZE。 '''
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p5')(P5)
    ''' 使用 MaxPooling2D 层处理 P5，得到 P6。 '''
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)
    ''' 定义 RPN 的特征图。 '''
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    ''' 定义 Mask R-CNN 的特征图。 '''
    mrcnn_feature_maps = [P2, P3, P4, P5]
    ''' 定义锚点。 '''
    anchors = input_anchors
    ''' 构建 RPN 模型。 '''
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    ''' 初始化 RPN 的分类和边界框预测。 '''
    rpn_class_logits, rpn_class, rpn_bbox = [], [], []
    ''' 对每个特征图进行 RPN 预测。 '''
    for i in rpn_feature_maps:
        logits, classes, bbox = rpn([i])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)
    ''' 合并 RPN 的分类预测结果。 '''
    rpn_class_logits = Concatenate(axis=1, name='rpn_class_logits')(rpn_class_logits)
    ''' 合并 RPN 的分类概率结果。 '''
    rpn_class = Concatenate(axis=1, name='rpn_probs')(rpn_class)
    ''' 合并 RPN 的边界框预测结果。 '''
    rpn_bbox = Concatenate(axis=1, name='rpn_bbox')(rpn_bbox)
    ''' 定义提议的数量。 '''
    proposal_count = config.POST_NMS_ROIS_INFERENCE
    ''' 使用 ProposalLayer 生成最终的 ROI。 '''
    rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD, name="ROI", config=config)([rpn_class, rpn_bbox, anchors])
    ''' 使用 FPN 分类器图生成分类和边界框预测。 '''
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta, config.POOL_SIZE, config.NUM_CLASSES, config.TRAIN_BN, config.FPN_CLASSIF_FC_LAYERS_SIZE)
    ''' 使用 DetectionLayer 生成最终检测结果。 '''
    detections = DetectionLayer(config, name='mrcnn_detection')([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

    ''' 提取检测框。 '''
    detection_boxes = Lambda(lambda x: x[..., :4])(detections)
    ''' 构建 FPN 掩码图。 '''
    mrcnn_mark = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps, input_image_meta, config.MASK_POOL_SIZE, config.NUM_CLASSES, config.TRAIN_BN)
    ''' 构建最终的 Mask R-CNN 模型。 '''
    model = Model([input_image, input_image_meta, input_anchors], [detections, mrcnn_class, mrcnn_bbox, mrcnn_mark, rpn_rois, rpn_class, rpn_bbox], name='mask_rcnn')
    return model
