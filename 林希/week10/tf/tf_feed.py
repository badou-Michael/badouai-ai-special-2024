import tensorflow as tf

# placeholder占位符，占一个内存空间的位置，现在不知道input1和input2将来会是什么样的数字，所以先占位
# 所以想传入什么值就可以传入什么值，不需要写死
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)   # 做乘法

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7], input2:[2.]}))

# 输出
# [array([14.], dtype=float32)]