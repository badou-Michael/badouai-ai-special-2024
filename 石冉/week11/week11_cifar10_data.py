#cifar10分为两部分：第一部分读取数据并预处理，第二部分构建网络结构并训练推理

#本文件是第一部分：读取cifar10数据并预处理
import os
import tensorflow as tf


#设定用于训练和推理的样本数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#定义一个空类，用于返回读取的cifar-10数据
class CIFAR10Record(object):
    pass

#定义一个读取cifar-10数据的函数read_cifar10
def read_cifar10(file_queue):
    #创建一个CIFAR10Record的实例
    result=CIFAR10Record()
    #计算元素数量=h*w*c(cifar10的图片尺寸都是32*32*3)+label_bytes(cifar10是1，cifar100是2)
    label_bytes=1
    result.height=32
    result.width=32
    result.depth=3
    #图片总元素数量是h*w*c
    image_bytes=result.height*result.width*result.depth
    #每个样本包含图片和标签，所以还要把标签数量也加上。cifar10是0-9个标签（1位），cifar100是0-99个标签（2位）
    record_bytes=image_bytes+label_bytes

    #使用tensorflow的FixedLengthRecordReader()读取文件,通过指定record_bytes，读取器知道每次应该读取多少字节的数据
    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
    #分别读取数据的index和value
    result.key,value=reader.read(file_queue)
    #将读取的字符串形式的数据转换为图像对应的像素素组
    record_bytes=tf.decode_raw(value,tf.uint8)

    #该数组的第一个元素是标签，把它提取出来，存储在result.label中
    #使用tf.strided_slice取recode_bytes的第0位开始，label_bytes位结束，将label提取出来。并用tf.cast转换为int32形式
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    #剩下的就是图片数据，仍然用strided_slice提取出来，但是要用reshape把c*h*w的一维数据转换为c,h,w三维数据
    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),
                           [result.depth,result.height,result.width])
    #将c,h,w转换为h,w,c
    result.uint8image=tf.transpose(depth_major,[1,2,0]) #原第0位放在第三位，原第1位放在第0位

    return result

#定义一个对读取数据进行预处理的函数
def inputs(data_dir,batch_size,distorted):
    #拼接地址
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]
    #创建一个字符串输入队列，这个队列用于管理文件名列表filenames中指定的文件
    file_queue=tf.train.string_input_producer(filenames)
    #用上面的read_cifar10函数读取数据
    read_input=read_cifar10(file_queue)
    #把图片转换为float32格式
    reshaped_image=tf.cast(read_input.uint8image,tf.float32)
    num_examples_per_epoch=num_examples_pre_epoch_for_train

    #下面对图片进行预处理，如果输入的distorted不为空，则进行预处理，否则不预处理
    if distorted != None:
        #对图片进行剪切
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])
        #对图片进行翻转
        flipped_image=tf.image.random_flip_left_right(cropped_image)
        #对图片进行亮度调整,max_delta表示能够进行亮度进行的最大随机调整程度
        brightened_image=tf.image.random_brightness(flipped_image,max_delta=0.8)
        #对图片进行对比度调整,lower是对比度因子的下限值，表示对比度调整的最小强度;
        # upper是对比度因子的上限值，表示对比度调整的最大强度
        contrast_image=tf.image.random_contrast(brightened_image,lower=0.2,upper=1.8)
        #标准化，对每个像素减去平均值再除以方差
        std_image=tf.image.per_image_standardization(contrast_image)

        #对图片和标签设置形状
        std_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        #队列管理
        min_queue_example=int(num_examples_pre_epoch_for_eval*0.4)
        print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.' % min_queue_example)

        #打乱训练图片的顺序，避免过拟合
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        images_train,labels_train=tf.train.shuffle_batch([std_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_example+3*batch_size,
                                                         min_after_dequeue=min_queue_example)

        return images_train,tf.reshape(labels_train,[batch_size])
    #其他情况下，不对图像进行增强处理，只进行reshape和标准化
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        std_image=tf.image.per_image_standardization(resized_image)
        #对图片和标签设置形状
        std_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch*0.4)
        #这里batch函数和前面的shuffle batch函数作用相同
        images_test,labels_test=tf.train.batch([std_image,read_input.label],
                                               batch_size=batch_size,num_threads=16,
                                               capacity=min_queue_examples+3*batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])
