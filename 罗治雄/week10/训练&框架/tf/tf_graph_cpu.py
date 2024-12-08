'''
在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU或 GPU). 
一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 
如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.
如果你的系统里有多个 GPU, 那么 ID 最小的 GPU 会默认使用。
'''
#如果你想要手动指派设备, 你可以用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.

import tensorflow as tf

# 新建一个graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b, name = "mul")
print("tesnor: ", c)
'''
如果你指定的设备不存在, 你会收到 InvalidArgumentError 错误提示。
可以在创建的 session 里把参数 allow_soft_placement 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
'''
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
print (sess.run(c))
sess.close()
