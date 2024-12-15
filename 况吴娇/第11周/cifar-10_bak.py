#该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev)) #stddev：标准差，用于初始化变量，控制随机初始化的范围。
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
#其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
#要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[batch_size])

#创建第一个卷积层 shape=(kh,kw,ci,co)
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0) #shape=[5, 5, 3, 64]：卷积核的形状，5x5的卷积核，3个输入通道，64个输出通道。
#w1=0.0：L2正则化的权重，这里设置为0，表示不使用正则化。
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64])) ##[ b1, b2, b3, ..., b64 ]
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME") #使用tf.nn.max_pool进行最大池化，池化窗口为3x3，步长为2。
'''
tf.nn.conv2d 函数的 strides 参数中，strides=[1,2,2,1] 表示卷积操作在每个维度上的步长。这个参数是一个长度为4的列表，分别对应于批次（batch）、高度（height）、宽度（width）和通道（channel）四个维度的步长。具体来说：

第一个 1 表示在批次维度上的步长为1，即卷积操作在不同批次的图像之间不跳过任何图像。
第二个 2 表示在高度维度上的步长为2，即卷积核在图像的高度方向上每次移动2个像素。
第三个 2 表示在宽度维度上的步长为2，即卷积核在图像的宽度方向上每次移动2个像素。
第四个 1 表示在通道维度上的步长为1，即卷积操作在图像的不同通道之间不跳过任何通道。
因此，strides=[1,2,2,1] 的设置意味着卷积核在图像的高度和宽度方向上每次移动2个像素，这将导致输出特征图的尺寸是输入特征图尺寸的一半（如果使用 "SAME" 填充方式）。这种设置在卷积神经网络中非常常见，因为它可以有效地减少特征图的尺寸，从而减少计算量和参数数量，同时保留重要的特征信息。
'''
#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME") #"SAME" 表示添加适当的填充以保证输出特征图的大小与输入特征图相同（或根据步长进行下采样）
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
'''
pool2：这是上一层（通常是池化层）的输出，通常是一个四维张量，形状为 [batch_size, height, width, channels]。
[batch_size, -1]：这是 tf.reshape 函数的目标形状参数。
batch_size：保持批次维度不变。
-1：这是一个特殊的参数，表示自动计算该维度的大小。具体来说，-1 会让 TensorFlow 自动计算出将 pool2 的剩余维度（height * width * channels）拉平为一维所需的大小。
这样，pool2 的三维结构（height, width, channels）被拉直为一个一维向量。
'''
dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值
'''
reshape.get_shape()：这个方法返回 reshape 张量的形状。由于 reshape 是一个二维张量，
其形状为 [batch_size, num_features]，其中 num_features 是拉平后的特征数量。
[1]：索引 1 用于获取形状的第二个维度，即拉平后的特征数量。
'''
#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)#加法（广播）：+ fc_bias1 将偏置向量加到矩阵乘法的结果上。
#ReLU激活函数：tf.nn.relu(...) 将应用于加法的结果。ReLU函数将所有负值置为0，保留正值不变。

#建立第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))
'''
tf.nn.sparse_softmax_cross_entropy_with_logits 是 TensorFlow 中的一个函数，用于计算 logits 和 labels 之间的稀疏 softmax 交叉熵。
这个函数主要用于离散分类任务，其中类别是互斥的（每个条目恰好属于一个类别）
此函数要求 labels 是一维向量，每个值是 [0, num_classes) 中的索引
'''

weights_with_l2_loss=tf.add_n(tf.get_collection("losses")) #使用tf.add_n(tf.get_collection("losses"))计算所有L2正则化损失的总和。
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

'''
tf.get_collection("losses")：

tf.get_collection 是 TensorFlow 中用于获取存储在特定集合中的所有项的函数。
这里的 "losses" 是一个集合的名称，之前在代码中通过 tf.add_to_collection("losses", weights_loss) 将各个权重的L2损失添加到了这个集合中。
因此，tf.get_collection("losses") 返回一个列表，包含了所有添加到 "losses" 集合中的损失项。
tf.add_n(tf.get_collection("losses"))：

tf.add_n 是 TensorFlow 中用于计算一个张量列表中所有元素的和的函数。
它接受一个张量列表作为输入，并返回这些张量的和。
在这里，tf.add_n(tf.get_collection("losses")) 计算所有收集在 "losses" 集合中的损失项的总和。

loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss：

这行代码将两个损失项相加，得到模型的总损失。
tf.reduce_mean(cross_entropy) 表示模型的平均交叉熵损失。
weights_with_l2_loss 表示模型的总L2正则化损失。
通过将这两个损失项相加，模型的总损失不仅考虑了预测误差（交叉熵损失），还考虑了模型复杂度（L2正则化损失）。
'''

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
'''
tf.train.AdamOptimizer(1e-3)：创建一个Adam优化器实例，Adam优化器是一种自适应学习率优化算法，通常在训练深度学习模型时表现良好。
.minimize(loss)：调用优化器的minimize方法，指定要最小化的损失函数（loss）。这个方法会自动计算损失函数关于模型参数的梯度，并更新参数以减少损失。
'''
#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(result,y_,1)
'''
计算模型输出的Top-K准确率，默认情况下计算Top-1准确率。
参数：
result：模型的输出，通常是一个softmax层的输出，表示每个类别的预测概率。
y_：真实的标签，与模型输出的形状和类型相匹配。
1：K值，表示要计算的Top-K准确率。这里设置为1，表示计算Top-1准确率，即预测概率最高的类别是否与真实标签一致。
过程：
tf.nn.in_top_k：这个函数返回一个布尔张量，表示每个样本的预测是否正确（即预测概率最高的类别是否与真实标签一致）。
'''
init_op=tf.global_variables_initializer()
#tf.global_variables_initializer()：这个函数返回一个操作，当执行这个操作时，会初始化图中定义的所有全局变量。这通常在训练开始前执行，以确保所有变量都被正确初始化。
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()      

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))
'''
for step in range(max_steps)：循环max_steps次，每次循环代表一个训练步骤。
start_time = time.time()：记录每个训练步骤开始的时间。
image_batch, label_batch = sess.run([images_train, labels_train])：从训练数据队列中获取一个批次的图像和标签。
_, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})：执行训练操作train_op并计算损失loss，feed_dict提供当前批次的数据。
duration = time.time() - start_time：计算每个训练步骤所需的时间。
if step % 100 == 0：每100步，输出训练信息。
examples_per_sec = batch_size / duration：计算每秒处理的样本数。
sec_per_batch = float(duration)：计算处理一个批次所需的时间。
print(...)：打印当前步骤、损失值、每秒处理的样本数和处理一个批次所需的时间。



num_batch = int(math.ceil(num_examples_for_eval / batch_size))：计算需要多少个批次来处理所有测试样本。
true_count = 0：初始化正确预测的样本计数。
total_sample_count = num_batch * batch_size：计算测试集中的总样本数。
for j in range(num_batch)：循环num_batch次，每次处理一个批次的测试数据。
image_batch, label_batch = sess.run([images_test, labels_test])：从测试数据队列中获取一个批次的图像和标签。
predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})：计算当前批次的预测结果，top_k_op返回预测正确的样本。
true_count += np.sum(predictions)：累加预测正确的样本数。
print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))：计算并打印模型的准确率。

这段代码首先执行训练循环，每100步输出一次训练信息，包括步骤、损失值、每秒处理的样本数和处理一个批次所需的时间。然后，它评估模型在测试集上的准确率，通过计算所有预测正确的样例个数，并打印最终的准确率。
'''