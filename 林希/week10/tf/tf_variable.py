import tensorflow as tf

#创建一个变量，初始化为标量为0
state = tf.Variable(0, name="counter")

#创建一个op，其作用是使 state 增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#启动图后，变量必须经过’初始化‘ (init) op 初始化，
#首先必须增加一个’初始化‘ op 到图中
init_op = tf.global_variables_initializer()

#启动图，运行 op
with tf.Session() as sess:
    #运行'init' op
    sess.run(init_op)
    #打印'state' 的初始值
    print("state", sess.run(state))
    #运行 op， 更新 'state'，并打印'state'
    for _ in range(5):
        sess.run(update)
        print("update:", sess.run(state))