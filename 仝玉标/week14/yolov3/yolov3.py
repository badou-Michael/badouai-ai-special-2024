
import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def detect(image_path, model_path, yolo_weights=None):
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入模型
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)

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

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            print(label_size)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        image.save('./img/result1.jpg')


if __name__ == '__main__':
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    else:
        detect(config.image_file, config.model_dir)
