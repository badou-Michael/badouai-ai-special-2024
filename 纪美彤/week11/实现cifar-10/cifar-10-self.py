import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
import math
import cifar10_data_self

max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="cifar_data/cifar-10-batches-bin"

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

# 用cifar10_data_self分别读取训练集和测试集
images_train,labels_train=cifar10_data_self.inputs_self(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=cifar10_data_self.inputs_self(data_dir=data_dir,batch_size=batch_size,distorted=None)

# 开始构建网络并训练
# 创建placeholder用于存放数据和标签,第一位是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y=tf.placeholder(tf.int32,[batch_size])

# 创建第一个卷积层，卷积核尺寸前两位认为设定，是一个卷积核的大小，第三位跟随输入图像的通道数，第四位影响输出的通道数
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
# tf自带卷积函数，stride有四个方向
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
# 偏置
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
# 激活函数
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
# 池化，最大池化，核尺寸：3*3
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#进行全连接操作之前将卷积的结果拍扁
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
dim=reshape.get_shape()[1]            #get_shape()[1].value表示获取reshape之后的第二个维度的值

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接层，最后一层不进行激活，而是接softmax
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))

# 把每个权重的loss和cross_entropy加起来，使loss和权重关联更密切，但也容易导致过拟合，且对网络帮助不大
weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

# 优化器
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(result,y,1)

# 定义初始化
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    #每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

    #计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))
