#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：detect.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 19:25 
@Desc : 
"""
from pathlib import Path

from typing import Optional

import fire
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fire import Fire
import tensorflow as tf

import config
from predict import YOLOv3Predictor
from utils import resize_image, load_weights

def detect(i: str, m: str, w: Optional[str]):
    """

    :param i: image filepath
    :param m: model filepath
    :param w: weights filepath
    :return:
    """
    img = Image.open(i)
    resized_img = resize_image(img, (416, 416))
    img_data = np.array(resized_img, dtype='float32')
    img_data /= 255.
    input_img_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_img = tf.placeholder(dtype=tf.float32, shape=[None, 416, 416, 3])


    predictor = YOLOv3Predictor()

    with tf.Session() as sess:
        if w is not None:
            weights_fp = Path(w)
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_img, input_img_shape)
                weights = load_weights(tf.global_variables('predict'), weight_fp=weights_fp)
                sess.run(weights)

        else:
            model_path = Path(m)
            boxes, scores, classes = predictor.predict(input_img, input_img_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path.as_posix())

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={input_img: img_data, input_img_shape: [img.size[1], img.size[0]]}
        )

        font = ImageFont.truetype(font=config.FONT_FILE,
                                  size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))

        thickness = (img.size[1] + img.size[0]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_cls = predictor.classes[c]
            box, score = out_boxes[i], out_scores[i]
            label = f"{predicted_cls} ({score:.2f})"
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(img.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(img.size[0] - 1, np.floor(right + 0.5).astype('int32'))
            if top - label_size[1] >= 0:
                text = np.array([left, top - label_size[1]])
            else:
                text = np.array([left, top + 1])

            map(lambda x: draw.rectangle((left + x, top + x, right - x, bottom - x)), range(thickness))

            draw.rectangle([tuple(text), tuple(text+label_size)], fill=predictor.colors[c])

            draw.text(text, label, font=font, fill=(0, 0, 0))
            del draw
        img.show()
        img.save(config.RESULT_IMG_FILE)


if __name__ == '__main__':
    fire.Fire(detect)
