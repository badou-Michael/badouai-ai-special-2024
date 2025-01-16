import os
import tensorflow as tf
num_classes=10


num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1                                           
    result.height=32
    result.width=32
    result.depth=3                                          

    image_bytes=result.height * result.width * result.depth 
    record_bytes=label_bytes + image_bytes                  

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)  
    result.key,value=reader.read(file_queue)                 

    record_bytes=tf.decode_raw(value,tf.uint8)              
    
    
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                           [result.depth,result.height,result.width])  

   
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result                               
def inputs(data_dir,batch_size,distorted):              
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]   

    file_queue=tf.train.string_input_producer(filenames)     
    read_input=read_cifar10(file_queue)                     
    reshaped_image=tf.cast(read_input.uint8image,tf.float32)   

    num_examples_per_epoch=num_examples_pre_epoch_for_train


    if distorted != None:                         #如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])          #首先将预处理好的图片进行剪切，使用tf.random_crop()函数

        flipped_image=tf.image.random_flip_left_right(cropped_image)    #将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)   #将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)    #将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数

        float_image=tf.image.per_image_standardization(adjusted_contrast)          #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24,24,3])                      #设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
                             #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train,tf.reshape(labels_train,[batch_size])

    else:                               #不对图像数据进行数据增强处理
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image=tf.image.per_image_standardization(resized_image)          #剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test,tf.reshape(labels_test,[batch_size])
