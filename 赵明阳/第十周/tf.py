import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
#使用numpy生成200个随机点
#np.linspace(-0.5, 0.5, 200) 生成在区间 [-0.5, 0.5] 上均匀分布的 200 个数据点，形成一个一维数组。然后通过 [:, np.newaxis]
# 操作将其扩展为二维数组（列向量形式，形状为 (200, 1)），这符合神经网络输入数据的常见格式要求（通常批量数据是二维的，每行代表
# 一个样本，每列代表一个特征）。
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
#np.random.normal(0, 0.02, x_data.shape) 按照均值为 0、标准差为 0.02 的正态分布生成与 x_data 形状相同的噪声数据，用于给目标
# 数据添加噪声，使模拟数据更贴近实际情况。
noise=np.random.normal(0,0.02,x_data.shape)
#y_data=np.square(x_data)+noise 生成目标数据，这里是将 x_data 中的每个元素进行平方操作（模拟一个二次函数关系），再加上对应的
# 噪声数据，得到带有噪声的目标输出 y_data，形状同样为 (200, 1)。
y_data=np.square(x_data)+noise
 
#定义两个placeholder存放输入数据
#在 TensorFlow 中，placeholder 用于定义在运行时传入数据的占位符。这里定义了两个占位符 x 和 y，数据类型都为 tf.float32
# （32 位浮点数）。
x=tf.placeholder(tf.float32,[None,1])
#[None, 1] 表示它们的形状，其中 None 表示在这个维度上可以传入任意数量的样本（也就是批量大小可以动态变化），而 1 表示每个样本
# 只有一个特征维度，符合前面生成的 x_data 和 y_data 的数据格式要求，用于在训练和预测时传入输入数据和对应的真实标签数据。
y=tf.placeholder(tf.float32,[None,1])
 
#定义神经网络中间层
#Weights_L1=tf.Variable(tf.random_normal([1, 10]))：定义中间层的权重变量，形状为 (1, 10)，表示将输入维度为 1 的数据映射到维度
# 为 10 的中间层表示，使用 tf.random_normal 函数进行随机正态分布初始化。
Weights_L1=tf.Variable(tf.random_normal([1,10]))
#biases_L1=tf.Variable(tf.zeros([1, 10]))：定义中间层的偏置变量，形状为 (1, 10)，初始值全部设为 0，偏置项用于在神经元的线性
# 变换基础上进行平移调整，增加模型的表达能力。
biases_L1=tf.Variable(tf.zeros([1,10]))    #加入偏置项
#Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1：通过 tf.matmul 函数将输入数据 x 与权重 Weights_L1 进行矩阵乘法运算，再加上
# 偏置 biases_L1，得到中间层神经元未经过激活函数的线性组合输出。
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
#L1=tf.nn.tanh(Wx_plus_b_L1)：使用 tf.nn.tanh 激活函数对中间层的线性输出进行非线性变换，tanh 函数将输入值映射到 (-1, 1)
# 区间内，引入非线性特性，使神经网络能够拟合更复杂的函数关系。
L1=tf.nn.tanh(Wx_plus_b_L1)   #加入激活函数

#定义神经网络输出层
#Weights_L2=tf.Variable(tf.random_normal([10, 1]))：定义输出层的权重变量，形状为 (10, 1)，用于将中间层维度为 10 的数据映射
# 到输出维度为 1 的预测结果，同样进行随机正态分布初始化。
Weights_L2=tf.Variable(tf.random_normal([10,1]))
#biases_L2=tf.Variable(tf.zeros([1, 1]))：定义输出层的偏置变量，形状为 (1, 1)，初始值设为 0。
biases_L2=tf.Variable(tf.zeros([1,1]))  #加入偏置项
#Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2：先将中间层输出 L1 与输出层权重 Weights_L2 进行矩阵乘法，
# 再加上偏置 biases_L2，得到输出层未经过激活函数的线性组合输出。
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
#prediction=tf.nn.tanh(Wx_plus_b_L2)：最后使用 tf.nn.tanh 激活函数对输出层的线性输出进行非线性变换，得到最终的预测结果
# prediction，其形状为 (?, 1)（? 表示批量大小由传入的数据决定）。
prediction=tf.nn.tanh(Wx_plus_b_L2)   #加入激活函数

#定义损失函数（均方差函数）
#tf.square(y-prediction) 计算真实标签 y 与预测结果 prediction 之间差值的平方，得到每个样本的误差平方。
#tf.reduce_mean 函数对所有样本的误差平方求平均值，得到一个标量值 loss，即均方差损失函数。它衡量了模型预测结果与真实数据之间的
# 平均误差程度，训练的目标就是要最小化这个损失值。
loss=tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）
#使用 tf.train.GradientDescentOptimizer(0.1) 创建一个梯度下降优化器，学习率设置为 0.1。学习率决定了每次更新模型参数时沿着梯度
# 方向移动的步长大小。然后通过 minimize(loss) 方法告诉优化器要最小化的目标是前面定义的损失函数 loss，优化器会根据损失函数对模型
# 参数（如权重和偏置变量）计算梯度，并按照梯度下降的方式更新参数，逐步降低损失值。
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
    #变量初始化：在 tf.Session() 上下文环境中，首先通过 sess.run(tf.global_variables_initializer()) 对所有定义的变量
    # （如权重和偏置变量）进行初始化，给它们赋予初始的随机值，这是在开始训练前必须要做的一步。
    sess.run(tf.global_variables_initializer())
    #训练2000次
    #每次迭代中通过 sess.run(train_step, feed_dict={x:x_data, y:y_data}) 执行一次优化器的 train_step 操作，
    # 传入当前批次的输入数据 x_data 和对应的真实标签 y_data，让优化器根据损失函数计算梯度并更新模型参数，逐步优化模型，
    # 使其能够更好地拟合数据。
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获取预测值与可视化：训练完成后，通过 sess.run(prediction, feed_dict={x:x_data}) 获取在输入 x_data 情况下模型的最终预测值
    # prediction_value。
 
    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
 
    #画图
    #使用matplotlib.pyplot库进行可视化，先绘制散点图展示真实的数据点（通过plt.scatter(x_data,y_data)），再绘制红色的曲线
    # 展示预测值（通过 plt.plot(x_data, prediction_value, 'r-', lw=5)，其中 'r-' 表示红色实线，lw=5 表示线宽为 5），
    # 最后通过 plt.show() 显示绘制好的图形，直观呈现模型对数据的拟合效果。
    plt.figure()
    plt.scatter(x_data,y_data)   #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
    plt.show()
