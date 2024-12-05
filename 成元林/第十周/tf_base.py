import tensorflow as tf

#定义变量
a1 = tf.Variable(0)
#定义常量
one = tf.constant(1)
newValue = tf.add(a1,one)
update = tf.assign(a1,newValue)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("a1",sess.run(a1))
    for i in range(4):
        sess.run(update)
        print("a1:",sess.run(a1))

