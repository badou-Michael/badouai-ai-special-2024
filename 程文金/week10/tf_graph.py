import tensorflow as tf

#构造器的返回值代表该常量op的返回值
matrix1 = tf.constant([[3.,3.]])
#创建另外一个常量op,产生一个2*1矩阵
matrix2 = tf.constant([[2.,2.]])

#创建一个矩阵乘法 matmul op,把'matrix1'和'matrix2'作为输入
product = tf.matmul(matrix1,matrix2)

#启动默认图
sess = tf.Session()

result = sess.run(product)
print(result)

#任务完成，关闭会话
sess.close()