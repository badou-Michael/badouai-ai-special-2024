import os
import tensorflow as tf
import numpy as np
import time
import math

num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class CIFAR10Record(object):
    pass

def readCifar10(file):
    result = CIFAR10Record()

    label_bytes = 1
    result.width = 32
    result.height = 32
    result.depth = 3

    image_bytes = result.depth*result.height*result.width
    record_bytes = label_bytes+image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file)
    record_bytes = tf.decode_raw(value,tf.uint8)

    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    image_data = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),
                            [result.depth,result.height,result.width])
    result.uint8image = tf.transpose(image_data,[1,2,0])
    return result

def inputs(data_dir,batch_size,distorded):
    fileName = [os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]
    file_queue = tf.train.string_input_producer(fileName)
    read_input = readCifar10(file_queue)

    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    if distorded != None:
        cropped_image = tf.random_crop(reshaped_image,[24,24,3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval*0.4)

        image_train,labels_train = tf.train.shuffle_batch([float_image,read_input.label],
                                                          batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity=min_queue_examples+batch_size*3,
                                                          min_after_dequeue = min_queue_examples)
        return image_train,tf.reshape(labels_train,[batch_size])
    else:
        resize_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)
        float_image = tf.image.per_image_standardization(resize_image)

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_train * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])



def variable_with_weight_loss(shape,std,w):
    var = tf.Variable(tf.truncated_normal(shape,stddev=std))
    if w is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var),w,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var


max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="cifar_data/cifar-10-batches-bin"

image_train,labels_train = inputs(data_dir,batch_size,True)
image_test,labels_test = inputs(data_dir,batch_size,None)

x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y = tf.placeholder(tf.int32,[batch_size])

kernel1 = variable_with_weight_loss([5,5,3,64],5e-2,0.0)
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

kernel2 = variable_with_weight_loss([5,5,64,64],5e-2,0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2 = tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

reshape = tf.reshape(pool2,shape=[batch_size,-1])
dim = reshape.get_shape()[1].value

weight1=variable_with_weight_loss([dim,384],0.04,0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

weight2=variable_with_weight_loss([384,192],0.04,0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
fc_2=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

weight3 = variable_with_weight_loss([192,10],1/192.0,0.0)
fc_bias3 = tf.Variable(tf.constant(0.1,shape=[10]))
fc_3 = tf.add(tf.matmul(fc_2,weight3),fc_bias3)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_3,labels=tf.cast(y,tf.int64))
weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy)+weight_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_op = tf.nn.in_top_k(fc_3,y,1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()

    for step in range (max_steps):
        start_time = time.time()
        image_batch,labels_batch = sess.run([image_train,labels_train])
        _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:labels_batch})
        duration = time.time()-start_time

        if step%100==0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (step, loss_value, examples_per_sec, sec_per_batch))


    num_batch = int(math.ceil(num_examples_for_eval/batch_size))
    true_count = 0
    for j in range(num_batch):

        image_batch,labels_batch = sess.run([image_test,labels_test])
        predictions = sess.run([top_op],feed_dict={x:image_batch,y:labels_batch})
        true_count += np.sum(predictions)

    print("accuracy= %.3f%%"%((true_count/num_examples_for_eval)*100))
