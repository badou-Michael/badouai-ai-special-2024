import os
import config
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from utils import load_weights, letterbox_image
from yolo_predict import yolo_predictor

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

# 加载模型，进行预测
def detect(image_path, model_path, yolo_weights = None):
    """
    Introduction
    ------------
        加载模型,进行预测
    Parameters
    ----------
        model_path: 模型保存路径, 当使用yolo_weights时无用
        image_path: 图片路径
    """
    #---------------------------------------#
    #   图片预处理
    #---------------------------------------#
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (config.input_shape, config.input_shape)) # 416
    image_data = np.array(resize_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis = 0)  # Add batch dimension.
    #---------------------------------------#
    #   图片输入
    #---------------------------------------#
    # 输入原图的size (必须是元组/列表)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    input_image = tf.placeholder(dtype = tf.float32, shape = [None, config.input_shape, config.input_shape, 3])
    # 进入yolo预测
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        # 注意: 无论是 if / else 都是预测
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入模型
            load_op = load_weights(tf.global_variables(scope = 'predict'),  weights_file = yolo_weights)
            print("load_op_len:", len(load_op))
            sess.run(load_op)

            # 进行预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict = {
                    input_image: image_data,
                    # 以 y、x 的方式传入
                    input_image_shape: [image.size[1], image.size[0]],
                }
            )
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict = {
                    input_image: image_data,
                    # 以 y、x 的方式传入
                    input_image_shape: [image.size[1], image.size[0]],
                }
            )
    print(out_boxes.shape, out_scores.shape, out_classes.shape)
    font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf', size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    # 厚度
    thickness = (image.size[0] + image.size[1]) // 300
    print("打印class:", out_classes)
    # 画框
    # 注: out_classes、out_boxes、out_scores 顺序一致
    # 一般按照 scores 从大到小排序, 要确保分数高的先画出来
    for i, c  in reversed(list(enumerate(out_classes))):
        # 获得预测名字、box、分数
        predicted_class = predictor.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        # 用于画框和文字
        draw = ImageDraw.Draw(image)
        # textsize用于获得写字的时候，按照这个字体，要多大的框
        label_size = draw.textsize(label, font)

        # 获得4个边 (上 左, 下 右)
        top, left, bottom, right = box
        # +0.5 是为了四舍五入, max0确保不会小于0 (即不超过上边界)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        # 确保不会超过下边界
        bottom = min(image.size[1] - 1,  np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0: 
            # 不超过图像的上边界时, 直接放在框的上面 (文本框底部与物体框顶部对齐)
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # 画框
        for i in range(thickness): # 通过thickness描边
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline = predictor.colors[c]
            )
        # 画label的背景框
        draw.rectangle(
            [tuple(text_origin),tuple(text_origin + label_size)],
            fill = predictor.colors[c]
        )
        # 指定位置以黑色写label
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    image.show()
    image.save('./img/result1.jpg')

if __name__ == '__main__':
    if config.pre_train_yolo3 == True: # 当使用yolo3自带权重时
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    else: # 当使用自训练模型的时候
        detect(config.image_file, config.model_dir)
