import tensorflow as tf

# TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点.
# 这个默认图对许多程序来说已经足够用了.
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])      # Tensor("Const:0", shape=(1, 2), dtype=float32)
# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)

'''
默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 
为了真正进行矩阵相乘运算, 并得到矩阵乘法的结果, 必须在会话里启动这个图.
启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
'''
# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
'''
session对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 “with” 代码块来自动完成关闭动作.

with tf.Session() as sess:
  result = sess.run([product])
  print (result)
'''