import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Preprocessing:

    def __init__(self, batch_size, src_data='train', enhance=False):
        #定义读取的数据
        self.src_Data = self.__data_load(src_data)

        # 定义几个基础值，因为已知cifar10的图片信息
        self.label_bytes = 1
        self.image_height = 32
        self.image_width = 32
        self.image_channels = 3

        # 定义一下图片数据
        image_bytes = self.image_height * self.image_width * self.image_channels  # 代表一张图有多少数据
        self.record_bytes = image_bytes + self.label_bytes  # 这里汇总了字节量，即加上了代表图片是哪个分类的数据

        # 建立数据管道，读取出labels和images，同时判断是否走一遍数据增强
        self.labels, self.images = self.data_read(batch_size, enhance)

    # 定义一个读取数据的类型
    def __data_load(self, src_data):
        support_types = {
            'train':[os.path.join('cifar_data/cifar-10-batches-bin/', f'data_batch_{i}.bin') for i in range(1, 6)],
            'test':['cifar_data/cifar-10-batches-bin/test_batch.bin']
        }
        return support_types[src_data]

    # 定义一个方法，用于读取数据
    def data_read(self, batch_size, enhance):
        # 创建数据管道pipeline
        queue = tf.train.string_input_producer(self.src_Data)

        # 配置pipeline每批次读取字节量
        reader = tf.FixedLengthRecordReader(record_bytes=self.record_bytes)  # 字节量

        # 通过管道和配置，读取出key和value
        batch_key, batch_value = reader.read(queue)

        # 对读取出的value进行解码，源数据是二进制字符串
        decoded_value = tf.decode_raw(
            batch_value,
            tf.uint8,  # 定义解码后的数据类型
        )

        '''对解码后的数据进行切片，分割出label和image'''
        # 先分割出label
        src_labels = tf.strided_slice(
            decoded_value,
            [0], [self.label_bytes],  # 起始和结束位
        )

        # 再分割出image
        src_images = tf.strided_slice(
            decoded_value,
            [self.label_bytes], [self.record_bytes],  # 起始和结束位
        )

        # labels转类型为int32方便后续使用
        labels = tf.cast(src_labels, tf.int32)

        # images转类型为float32并reshape为图形形状方便后续使用
        images = tf.reshape(
            tf.cast(src_images, tf.float32),
            [self.image_height, self.image_width, self.image_channels]
        )

        # 判断是否需要数据增强
        if enhance:
            enhance_labels, enhance_images = self.__data_enhance(labels, images, batch_size)
            return enhance_labels, enhance_images
        else:
            images = tf.image.resize_image_with_crop_or_pad(images, 24, 24)
            return labels, images

    # 定义一个方法，用于数据增强
    def __data_enhance(self, labels, images, batch_size):
        # 随机剪裁
        cropped_images = tf.random_crop(
            images,
            [24, 24, 3],
        )

        # 随机水平翻转
        flipped_images = tf.image.random_flip_left_right(
            cropped_images,
        )

        # 随机亮度
        brightness_images = tf.image.random_brightness(
            flipped_images,
            0.8,
        )

        # 随机对比度
        contrasted_images = tf.image.random_contrast(
            brightness_images,
            0.2,
            1.8,
        )

        # 标准化
        adjusted_images = tf.image.per_image_standardization(contrasted_images)

        # 通过显式调用让图片确定在24*24*3的形状上
        adjusted_images.set_shape([24, 24, 3])
        labels.set_shape([1])

        '''
        打乱顺序        
        
	    多个线程（由 num_threads 参数指定）并行地将数据填充到队列中。只要队列中的元素数量小于 capacity，这些线程就可以继续添加数据。
	    当队列中的元素数量达到或超过 min_after_dequeue 时，系统会从队列中随机抽取一批（batch）数据。这意味着只有当队列中有足够多的元素时，才会进行随机抽取，以确保良好的随机性。
	    每次抽取一个批次的数据后，队列中的元素数量会减少。如果队列中的元素数量仍然大于 min_after_dequeue，则可以继续抽取；否则，系统会等待，直到队列中有足够的元素可以被打乱。
	    如果队列中的元素数量达到了 capacity，所有试图向队列添加新元素的操作将会被阻塞，直到队列中有空间可用。
	    如果队列中的元素数量低于 min_after_dequeue，所有试图从队列中抽取批次的操作将会被阻塞，直到队列中有足够的元素可以被打乱。
	    
	    通常，capacity 应该设置为 min_after_dequeue + (num_threads + 小余量) * batch_size
	    通常，min_after_dequeue 应该设置为一个合理的值，既能保证良好的随机性，又不会过度消耗资源。一个常见的做法是将其设置为 batch_size 的若干倍（如 10 倍）。
	    
	    所以最终，labels是以100行为一个输出
        '''
        labels, images = tf.train.shuffle_batch(
            [labels, adjusted_images],
            batch_size=batch_size,
            capacity=batch_size * 20 + (16 + 4) * batch_size,
            min_after_dequeue=batch_size * 20,
            num_threads=16,
        )

        # 将labels reshape成后续能够读取的一维张量，images可以不用管
        return tf.reshape(labels, [batch_size]), images


if __name__ == '__main__':
    q = Preprocessing(batch_size=100, enhance=True, src_data='train')
    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        sess.run(q.images)









