# yolo3检测图像

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import tensorflow as tf

from yolo3.utils import letterbox_image, load_weights
from yolo3 import config
from yolo3.yolo_predict import yolo_predictor

def detect(image_path, model_path, yolo_weights = None):
    """
    加载模型，进行预测
    :param image_path: 图片路径
    :param model_path: 模型路径
    :param yolo_weights: 权重路径(空的话不使用，自训练获得)
    :return: 图片预测结果
    """

    # 图片预处理
    image = Image.open(image_path)  # 打开图片
    resize_image = letterbox_image(image, (416, 416))  # 按照长宽比进行缩放
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.  # 归一化
    image_data = np.expand_dims(image_data, axis=0)

    input_image_shape = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.compat.v1.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 进行预测
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)

    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)

            # 载入模型
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)

            # 进行预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    # 传入的是[height, width]
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

        # 画检测框
        print(f'Found {len(out_boxes)} boxes for image.')

        # 设置字体
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300  # 画框的厚度

        draw = ImageDraw.Draw(image)

        for i, c in reversed(list(enumerate(out_classes))):
            # 获取预测的类别名、框和分数
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 设置标签格式
            label = f'{predicted_class} {score:.2f}'

            # 获取文本大小
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            # 对框的位置进行修正，确保框不会超出图像边界
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))

            # 打印框的信息
            print(label, (left, top), (right, bottom))
            print(label_size)

            # 设置文本的位置，避免文字超出图像
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 绘制边框
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c])  # 使用类别对应的颜色

            # 绘制标签背景
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c])  # 使用类别对应的颜色

            # 绘制标签文本
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw  # 清理draw对象

        # 显示和保存图像
        image.show()
        image.save('./img/result1.jpg')


if __name__ == '__main__':
    if config.pre_train_yolo3 == True: # 使用预先设定的weights进行推理
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    else:  # 重新训练模型
        detect(config.image_file, config.model_dir)