#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：convolution_neural_network.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2024/12/12 12:43 
@Desc : cnn impl
"""
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow_core import FIFOQueue

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR.joinpath('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

slim = tf.contrib.slim


class BaseCNN:
    name: str = None

    def __init__(self, *args, **kwargs):
        if self.name is None:
            raise ValueError(f"{self.name} is not defined")
        self.data_dir = DATA_DIR.joinpath(self.name)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def process_data(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def train_and_evaluate(self):
        raise NotImplementedError


class CifarRecord:
    label_bytes: int = 1
    height: int = 32
    width: int = 32
    channels: int = 3
    image_bytes: int = height * width * channels
    total_bytes: int = image_bytes + label_bytes
    label = tf.Tensor
    data = tf.Tensor


class Cifar10Network(BaseCNN):
    name = 'cifar'

    def __init__(
            self,
            train_sample: int = 50000,
            test_sample: int = 10000,
            batch_size: int = 100,
            epoch: int = 4000,
    ):

        super().__init__()

        self.train_sample = train_sample
        self.test_sample = test_sample
        self.batch_size = batch_size
        self.epoch = epoch
        self.output_shape = [24, 24, 3]
        self.data = tf.placeholder(tf.float32, shape=[self.batch_size, *self.output_shape])
        self.label = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.op, self.loss, self.top_k = None, None, None
        self.build()

    @staticmethod
    def __extract_data(queue: FIFOQueue) -> CifarRecord:
        record = CifarRecord()
        key, value = tf.FixedLengthRecordReader(record_bytes=record.total_bytes).read(queue)
        value = tf.decode_raw(value, tf.uint8)

        label = tf.strided_slice(value, [0], [record.label_bytes])
        data = tf.strided_slice(value, [record.label_bytes], [record.total_bytes])

        label = tf.cast(label, tf.int32)
        record.label = label

        data = tf.reshape(data, [record.channels, record.height, record.width])
        tf.transpose(data, perm=[1, 2, 0])
        tf.cast(data, tf.float32)
        record.data = data
        return record

    @staticmethod
    def __calc_weight_loss(shape: List[int], stddev: float = 5e-2, weight: float = .0) -> tf.Variable:
        variable = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        if weight is not None:
            loss = tf.multiply(tf.nn.l2_loss(variable), weight, name="loss")
            tf.add_to_collection("loss", loss)
        return variable

    def process_data(self, is_distorted: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        flies = list(map(lambda x: x.as_posix(), self.data_dir.glob('*.bin')))
        file_queue = tf.train.string_input_producer(flies)
        record = self.__extract_data(file_queue)

        if is_distorted:
            cropped = tf.random_crop(record.data, self.output_shape)
            bright = tf.image.random_brightness(cropped, max_delta=0.8)
            contrasted = tf.image.random_contrast(bright, lower=0.2, upper=1.8)
            standard = tf.image.per_image_standardization(contrasted)

        else:
            cropped = tf.image.resize_image_with_crop_or_pad(record.data, *self.output_shape[:-1])
            standard = tf.image.per_image_standardization(cropped)

        standard.set_shape(self.output_shape)
        record.label.set_shape([1])

        data, label = tf.train.shuffle_batch(
            [standard, record.label],
            batch_size=self.batch_size,
            num_threads=16,
            capacity=self.batch_size * 3 + int(self.train_sample * 0.4),
            min_after_dequeue=int(self.train_sample * 0.4),
        )
        label = tf.reshape(label, [self.batch_size])
        return data, label

    def build(self):
        # conv1
        kernel1 = self.__calc_weight_loss([5, 5, 3, 64])
        conv1 = tf.nn.conv2d(self.data, kernel1, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[64]))
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias))
        pool1 = tf.nn.max_pool(relu1, [1, 3, 3, 1], padding='SAME', strides=[1, 2, 2, 1])

        # conv2
        kernel2 = self.__calc_weight_loss([5, 5, 64, 64])
        conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
        pool2 = tf.nn.max_pool(relu2, [1, 3, 3, 1], padding='SAME', strides=[1, 2, 2, 1])

        # flatten
        flatten = pool2.reshape(pool2, [self.batch_size, -1])
        dim = flatten.get_shap()[1].value

        # fc1
        w1 = self.__calc_weight_loss([dim, 384], .04, .004)
        wb1 = tf.Variable(tf.constant(0.1, shape=[384]))
        fc1 = tf.nn.relu(tf.matmul(flatten, w1) + wb1)

        # fc2
        w2 = self.__calc_weight_loss([384, 192], .04, .004)
        wb2 = tf.Variable(tf.constant(0.1, shape=[192]))
        fc2 = tf.nn.relu(tf.matmul(fc1, w2) + wb2)

        # fc3
        w2 = self.__calc_weight_loss([192, 10], 1.0 / 192, .0)
        wb3 = tf.Variable(tf.constant(0.1, shape=[10]))
        fc3 = tf.nn.relu(tf.matmul(fc2, w2) + wb3)

        # calc cross_entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=tf.cast(self.label, tf.int64))

        # calc loss
        loss = tf.add_n(tf.get_collection("loss"))
        self.loss = tf.reduce_mean(cross_entropy) + loss

        # optimizer
        self.op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        # top_k
        self.top_k = tf.nn.top_k(fc3, self.label, 1)

    def train_and_evaluate(self):

        train_data, train_label = self.process_data()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(ses=sess)

        for i in range(self.epoch):
            data_batch, label_batch = sess.run([train_data, train_label])
            _, loss, val = sess.run([self.op, self.loss], feed_dict={self.data: data_batch, self.label: label_batch})

        test_data, test_label = self.process_data(False)

        true_count = 0
        for i in range(math.ceil(self.test_sample / self.batch_size)):
            data, label = sess.run([test_data, test_label])
            prediction = sess.run([self.top_k], feed_dict={self.data: data, self.label: label})
            true_count += np.sum(prediction)

        accuracy = true_count / self.test_sample

        print(f"{self.name} model's accuracy is {accuracy}")


class AlexNetwork(BaseCNN):
    name = "alex"

    def __init__(self,
                 input_shape: List[int] = None,
                 output_shape: int = 2,
                 batch_size: int = 128
                 ):

        super().__init__()

        if input_shape is None:
            self.input_shape = [224, 224, 3]

        self.output_shape = output_shape
        self.batch_size = batch_size
        self.model = self.build()

    @staticmethod
    def resize_img(img: np.ndarray, size: List[int]) -> np.ndarray:
        with tf.name_scope("resize_img"):
            images = [cv2.resize(i, size) for i in img]
            return np.array(images)

    def generate_array_from_lines(self, lines: List[str]) -> np.ndarray:
        total, i = len(lines), 0
        while 1:
            x_train, y_train = [], []
            for _ in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(lines)
                name = lines[i].split(";")[0]
                img = cv2.imread(self.data_dir.joinpath("image", "train", name).as_posix(), 0)
                img /= 255
                x_train.append(img)
                y_train.append(lines[i].split(";")[1])
                i = (i + 1) % total

            x_train = self.resize_img(np.array(x_train), self.input_shape)
            x_train.reshape(-1, 224, 224, 3)
            y_train = np_utils.to_categorical(np.array(y_train), num_classes=2)
            yield x_train, y_train

    def process_data(self) -> Tuple[List[str], int, int]:

        filepath = self.data_dir.joinpath("dataset.text")
        with open(filepath, "r") as f:
            lines = f.readlines()

        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

        test = int(len(lines) * 0.1)
        train = len(lines) - test
        return lines, train, test

    def build(self) -> Sequential:
        model = Sequential()
        # cov1
        model.add(
            Conv2D(48, (11, 11), (4, 4), activation='relu', input_shape=(self.input_shape,))
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))

        # conv2
        model.add(
            Conv2D(128, (5, 5), (1, 1), activation='relu', padding='same')
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))

        # conv3
        model.add(
            Conv2D(192, (3, 3), (1, 1), activation='relu', padding='same')
        )
        model.add(BatchNormalization())

        # conv4
        model.add(
            Conv2D(128, (3, 3), (1, 1), activation='relu', padding='same')
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))

        # flatten
        model.add(Flatten())

        # fc1
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))

        # fc2
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(self.output_shape, activation='softmax'))
        return model

    def train_and_evaluate(self):
        lines, train, test = self.process_data()
        period = ModelCheckpoint(
            self.data_dir.joinpath('ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5').as_posix(),
            monitor='acc',
            save_weights_only=False,
            save_best_only=False,
            period=3,
        )

        lr_reduce = ReduceLROnPlateau(
            monitor='acc',
            factor=0.5,
            patience=3,
            verbose=1,
        )

        stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1,
        )

        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-3), metrics=["accuracy"])

        self.model.fit_generator(
            generator=self.generate_array_from_lines(lines[:train]),
            steps_per_epoch=max(1, train // self.batch_size),
            validation_data=self.generate_array_from_lines(lines[train:train + test]),
            epochs=50,
            initial_epoch=0,
            callbacks=[period, lr_reduce, stopping],
        )
        self.model.save_weights(self.data_dir.joinpath('last1.h5').as_posix())

    def predict(self):
        filepath = self.data_dir.joinpath("last1.h5")
        self.model.load_weights(filepath.as_posix())
        filepath = self.data_dir.joinpath("test2.jpg")
        img = cv2.imread(filepath.as_posix(), 0) / 255
        img = np.expand_dims(img, axis=0)
        self.resize_img(img, self.input_shape[:-1])
        self.model.predict(img)


class VGG16Network(BaseCNN):
    name = "vgg16"

    def __init__(
            self,
            outputs: int = 1000,
            is_training: bool = True,
            dropout_keep_prob: float = 0.5,
            spatial_squeeze: bool = True,
            scope: str = "vgg16",
    ):
        super().__init__()

        self.outputs = outputs
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze
        self.scope = scope
        self.inputs = None

    @staticmethod
    def load_image(fp: Path):
        img = mpimg.imread(fp.as_posix())
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        return crop_img

    @staticmethod
    def resize_image(
            image: np.ndarray,
            size: Tuple[int, int],
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners: bool = False
    ):
        with tf.name_scope('resize_image'):
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, size,
                                           method, align_corners)
            image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
            return image

    def process_data(self):
        pass

    def build(self):
        with tf.variable_scope(self.scope, "vgg16", [self.inputs]):
            model = slim.repeat(self.inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
            model = slim.max_pool2d(model, [2, 2], scope="pool1")
            model = slim.repeat(model, 2, slim.conv2d, 128, [3, 3], scope="conv2")
            model = slim.max_pool2d(model, [2, 2], scope="pool2")
            model = slim.repeat(model, 3, slim.conv2d, 256, [3, 3], scope="conv3")
            model = slim.max_pool2d(model, [2, 2], scope="pool3")
            model = slim.repeat(model, 3, slim.conv2d, 512, [3, 3], scope="conv4")
            model = slim.max_pool2d(model, [2, 2], scope="pool4")
            model = slim.repeat(model, 3, slim.conv2d, 512, [3, 3], scope="conv5")
            model = slim.max_pool2d(model, [2, 2], scope="pool5")
            model = slim.conv2d(model, 4096, [7, 7], padding="VALID", scope="fc6")
            model = slim.dropout(model, dropout_keep_prob=self.dropout_keep_prob, is_training=self.is_training,
                                 scope="dropout6")
            model = slim.conv2d(model, 4096, [1, 1], scope="fc7")
            model = slim.dropout(model, dropout_keep_prob=self.dropout_keep_prob, is_training=self.is_training,
                                 scope="dropout7")
            model = slim.conv2d(model, self.outputs, [1, 1], scope="fc8", activation_fn=None, normalizer_fn=None)
            if self.spatial_squeeze:
                model = tf.squeeze(model, [1, 2], name="fc8/spatial_squeeze")

            return model

    def train_and_evaluate(self):
        pass

    def predict(self):
        filepath = self.data_dir.joinpath("table.jpg")
        img = self.load_image(filepath)
        inputs = self.resize_image(img, (224, 224))
        self.inputs = inputs
        prediction = self.build()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tf.train.Saver().restore(session, self.data_dir.joinpath('vgg_16.ckpt').as_posix())
            pro = tf.nn.softmax(prediction)
            pre = session.run(pro, feed_dict={tf.placeholder(tf.float32, [None, None, 3]): img})
            print(pre[0])


if __name__ == '__main__':
    Cifar10Network().train_and_evaluate()
    AlexNetwork().train_and_evaluate()
    VGG16Network().predict()
