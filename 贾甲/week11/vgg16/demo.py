#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-12-10
from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

img1 = utils.load_image('./test_data/table.jpg')

inputs = tf.placeholder(tf.float32,[None,None,3])
resuzed_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg_16(resized_img)

sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore

pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

print()
utils.print_prob()


