# -*- coding: utf-8 -*-
# time: 2024/11/26 16:18
# file: detect.py
# author: flame
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
import config
from utils import letterbox_image, load_weights
from yolo_predict import yolo_predictor

# 设置可见的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

''' 
此函数用于检测图像中的物体。它首先加载图像并将其调整到适合模型输入的大小，然后根据提供的模型路径和权重文件进行预测，
并在图像中标记出检测到的物体。最后，显示并保存带有标记的图像。
'''
def detect(image_path, model_path, yolo_weights = None):
    ''' 打开图像文件。 '''
    image = Image.open(image_path)

    ''' 将图像调整到416x416的大小，以适应模型输入要求。 '''
    resize_image = letterbox_image(image, (416,416))

    ''' 将调整后的图像转换为浮点数数组。 '''
    image_data = np.array(resize_image, dtype=np.float32)

    ''' 将图像数据归一化到0-1之间。 '''
    image_data /= 255.

    ''' 增加一个维度，使其成为批次数据。 '''
    image_data = np.expand_dims(image_data, 0)

    ''' 定义一个占位符，用于存储图像的原始尺寸。 '''
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))

    ''' 定义一个占位符，用于存储输入图像数据。 '''
    input_image = tf.placeholder(shape=[None, 416,416,3], dtype=tf.float32)

    ''' 创建YOLO预测器对象，用于进行目标检测。 '''
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)

    ''' 使用 TensorFlow 会话进行预测。 '''
    with tf.Session() as sess:
        ''' 如果提供了 YOLO 权重文件，则加载权重并进行预测。 '''
        if yolo_weights is not None:
            ''' 在 'predict' 作用域下定义预测操作。 '''
            with tf.variable_scope('predict'):
                ''' 获取预测的边界框、得分和类别。 '''
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)

                ''' 加载 YOLO 权重文件。 '''
                load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)

                ''' 运行加载操作。 '''
                sess.run(load_op)

                ''' 运行预测操作，获取输出的边界框、得分和类别。 '''
                out_boxes, out_scores, out_classes = sess.run(
                    [boxes, scores, classes],
                    feed_dict={input_image: image_data, input_image_shape: [image.size[1], image.size[0]]}
                )
        # 如果未提供 YOLO 权重文件，则从模型路径加载模型并进行预测。
        else:
            ''' 获取预测的边界框、得分和类别。 '''
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)

            ''' 创建一个 Saver 对象，用于恢复模型。 '''
            saver = tf.train.Saver()

            ''' 从模型路径恢复模型。 '''
            saver.restore(sess, model_path)

            ''' 运行预测操作，获取输出的边界框、得分和类别。 '''
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={input_image: image_data, input_image_shape: [image.size[1], image.size[0]]}
            )

        ''' 打印检测到的边界框数量和图像名称。 '''
        print("{} boxes , name: {}".format(len(out_boxes), 'img'))

        ''' 加载字体文件，用于在图像上绘制文本。 '''
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        ''' 计算绘制边框的厚度。 '''
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // 300

        ''' 遍历检测到的每个边界框。 '''
        for i, c in reversed(list(enumerate(out_classes))):
            ''' 获取预测的类别名称。 '''
            predicted_class = predictor.class_names[int(c)]

            ''' 获取预测的得分。 '''
            score = out_scores[i]

            ''' 获取预测的边界框坐标。 '''
            box = out_boxes[i]

            ''' 构建标签字符串，包含类别名称和得分。 '''
            label = '{} {:.2f}'.format(predicted_class, score)

            ''' 创建一个绘图对象，用于在图像上绘制。 '''
            draw = ImageDraw.Draw(image)

            ''' 计算标签文本的大小。 '''
            label_size = draw.textsize(label, font)

            ''' 获取边界框的坐标。 '''
            top, left, bottom, right = box

            ''' 确保边界框的坐标在图像范围内。 '''
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))

            ''' 打印标签和边界框坐标。 '''
            print(label, (left, top), (right, bottom))

            ''' 打印标签文本的大小。 '''
            print(label_size)

            ''' 计算标签文本的起始位置。 '''
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            ''' 绘制边界框。 '''
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[int(c)])

            ''' 绘制标签背景。 '''
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=predictor.colors[int(c)])

            ''' 在图像上绘制标签文本。 '''
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            ''' 删除绘图对象。 '''
            del draw

        ''' 显示带有标记的图像。 '''
        image.show()

        ''' 保存带有标记的图像。 '''
        image.save('./img/result2.jpg')

if __name__ == '__main__':
    ''' 根据配置文件决定是否使用预训练的 YOLO3 权重文件。 '''
    if config.pre_train_yolo3 == True:
        ''' 调用 detect 函数，传入图像文件路径、模型路径和 YOLO3 权重文件路径。 '''
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    else:
        ''' 调用 detect 函数，传入图像文件路径和模型路径。 '''
        detect(config.image_file, config.model_dir)
