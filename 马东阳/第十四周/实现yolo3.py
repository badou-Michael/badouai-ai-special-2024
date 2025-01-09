# yolo3
'''
YOLO（You Only Look Once）系列算法是目标检测领域中具有重要影响力的实时目标检测算法。YOLOv3作为该系列的重要版本，在准确性和速度方面都取得了显著的提升。

（一）网络结构
1. Darknet - 53骨干网络- YOLOv3采用Darknet - 53作为骨干网络，它由一系列的卷积层和残差块组成。这种结构能够有效地提取图像的特征，同时减少网络参数，避
免过拟合。 - 残差块的设计使得网络能够学习到更深层次的特征，即使在网络层数增加的情况下，也能保持较好的性能。例如，在处理复杂场景的图像时，残差块可以更好地捕
捉到物体的细节特征和语义信息。
2. 多尺度预测 - YOLOv3在三个不同尺度上进行预测，分别对应于网络的不同层。这种多尺度预测的方式能够检测到不同大小的目标。 - 例如，对于小目标检测，网络
在较深层的特征图上进行预测，因为深层特征图具有较高的语义信息但分辨率较低；而对于大目标检测，在较浅层的特征图上进行预测，浅层特征图分辨率高但语义信息相对较少。
通过这种方式，YOLOv3能够平衡目标的大小和语义信息，提高检测的准确性。
（二）预测原理
1. 边界框预测 - YOLOv3预测每个边界框的位置和尺寸信息。对于每个网格单元，它预测多个边界框，每个边界框包含坐标信息（中心坐标和宽高）以及置信度得分。
 - 置信度得分反映了该边界框包含目标的可能性以及预测的准确性。例如，在检测行人时，如果一个边界框的置信度得分较高，说明模型认为这个边界框很可能包含一个行人，
 并且预测的位置和尺寸比较准确。
2. 类别预测 - 同时，YOLOv3还对每个边界框预测其所属的类别。它采用多标签分类的方法，即一个边界框可以属于多个类别（在某些数据集上可能存在这种情况）。 
- 例如，在一个包含汽车和交通标志的图像中，一个边界框可能既被预测为汽车类别，又因为其包含部分交通标志的信息而被预测为交通标志类别（如果允许多标签分类）。
（三）损失函数
1. 位置损失 - 位置损失主要用于衡量预测边界框与真实边界框之间的位置差异。采用均方误差（MSE）等方法来计算位置损失，激励模型准确预测目标的位置。 
- 例如，如果预测的边界框中心坐标与真实坐标相差较大，位置损失就会增大，促使模型在训练过程中调整参数以减小这种差异。
2. 置信度损失 - 置信度损失用于衡量预测边界框的置信度得分与真实情况之间的差异。对于包含目标的边界框和不包含目标的边界框，采用不同的计算方法来
计算置信度损失。 - 对于包含目标的边界框，希望置信度得分尽可能接近1；对于不包含目标的边界框，希望置信度得分尽可能接近0。
3. 类别损失 - 类别损失用于衡量预测类别与真实类别之间的差异。采用交叉熵损失等方法来计算类别损失，促使模型正确预测目标的类别。 - 如果模型错误
地将一个行人预测为汽车类别，类别损失就会增大，引导模型在训练过程中学习正确的类别分类。

'''


import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw



def load_weights(var_list, weights_file):
    """
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
        var_list: 赋值变量名
        weights_file: 权重文件
    Returns
    -------
        assign_ops: 赋值更新操作
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv2d' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def letterbox_image(image, size):
    """
    Introduction
        对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    Parameters
        image: 输入图像
        size: 图像大小
    Returns
        boxed_image: 缩放后的图像
    """
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def detect(image_path, model_path, yolo_weights = None):
    """
    Introduction
        加载模型，进行预测
    Parameters
        model_path: 模型路径，当使用yolo_weights无用
        image_path: 图片路径
    """
    # ---------------------------------------#
    #   图片预处理
    # ---------------------------------------#
    image = Image.open(image_path)
    # 对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype = np.float32)
    # 归一化
    image_data /= 255.
    # 转格式，第一维度填充
    image_data = np.expand_dims(image_data, axis = 0)
    # ---------------------------------------#
    #   图片输入
    # ---------------------------------------#
    # input_image_shape原图的size
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    # 图像
    input_image = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)

    # 进入yolo_predictor进行预测，yolo_predictor是用于预测的一个对象
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        # ---------------------------------------#
        #   图片预测
        # ---------------------------------------#
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入模型
            load_op = load_weights(tf.global_variables(scope = 'predict'), weights_file = yolo_weights)
            sess.run(load_op)

            # 进行预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    # image_data这个resize过
                    input_image: image_data,
                    # 以y、x的方式传入
                    input_image_shape: [image.size[1], image.size[0]]
                })
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })

        # ---------------------------------------#
        #   画框
        # ---------------------------------------#
        # 找到几个box，打印
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font = 'CV1/font/FiraMono-Medium.otf', size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        # 厚度
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            # 获得预测名字，box和分数
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 打印
            label = '{} {:.2f}'.format(predicted_class, score)

            # 用于画框框和文字
            draw = ImageDraw.Draw(image)
            # textsize用于获得写字的时候，按照这个字体，要多大的框
            label_size = draw.textsize(label, font)

            # 获得四个边
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1 ] -1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0 ] -1, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            print(label_size)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline = predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill = predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        image.save('./yolo3_img/result1.jpg')

if __name__ == '__main__':

    # 当使用yolo3自带的weights的时候
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)

    # 当使用自训练模型的时候
    else:
        detect(config.image_file, config.model_dir)
