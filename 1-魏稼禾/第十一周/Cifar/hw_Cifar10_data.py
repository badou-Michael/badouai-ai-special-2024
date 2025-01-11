import os
import tensorflow as tf
num_classes=10

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class CIFAR10Record(object):
    pass


# 读取cifar10的文件队列
def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1
    result.height=32
    result.width=32
    result.depth=3

    image_bytes=result.height * result.width * result.depth
    record_bytes=label_bytes + image_bytes
    
    #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value=reader.read(file_queue)

    record_bytes=tf.decode_raw(value,tf.uint8)
    
    #因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    #剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                           [result.depth,result.height,result.width])  

    #转换数据排布方式，变为(h,w,c)
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result

#这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir,batch_size,distorted):
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)] #拼接地址

    file_queue=tf.train.string_input_producer(filenames)
    read_input=read_cifar10(file_queue)

    reshaped_image=tf.cast(read_input.uint8image,tf.float32)    #将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch=num_examples_pre_epoch_for_train


    if distorted != None:
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])
        flipped_image=tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)
        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)
        float_image=tf.image.per_image_standardization(adjusted_contrast) # 归一化，减去平均值，除以方差

        float_image.set_shape([24,24,3])    #设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)
        
        #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )

        return images_train,tf.reshape(labels_train,[batch_size])

    else:
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)  #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image=tf.image.per_image_standardization(resized_image)   #剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        #这里使用batch()函数代替tf.train.shuffle_batch()函数
        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])
