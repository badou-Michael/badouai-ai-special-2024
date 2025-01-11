# -*- coding: utf-8 -*-
# time: 2024/11/8 16:07
# file: NeturalNetWork_init.py
# author: flame
import numpy
import scipy.special


class NeturalNetWork:
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        ''' 初始化网络 设置输入层 输出层 和节点数 '''
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # 设置学习率
        self.lR = learningRate

        '''
        初始化权重矩阵吗，有两个权重矩阵 1 wih 输入层 -> 中间层 链路权重形成的矩阵 2 who 中间层 -> 输出层 链路形成的矩阵'''
        self.wih = numpy.random.rand(self.hNodes,self.iNodes) - 0.5
        self.who = numpy.random.rand(self.oNodes,self.hNodes)

        '''
        设置激活函数为sigmoid
        选择sigmoid的原因是它在神经网络中常用，能够将输入映射到(0,1)之间，适合作为输出层的激活函数。
        此外，sigmoid函数在处理二分类问题时特别有效，因为它可以将输入压缩到0和1之间，代表概率。
        使用lambda表达式的原因是它提供了一种简洁的方式来定义单行的小型匿名函数。
        这里我们使用lambda表达式来定义一个简单的函数，该函数接受一个参数x，并返回scipy.special.expit(x)的结果。
        lambda表达式使得代码更加简洁，特别是在不需要多次复用该函数的情况下。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self):
        pass
    
    def query(self, inputs):
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih,inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层接收到的信号量
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # 计算输出层经过激活函数输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

'''尝试输入一些数据对搭建的神经网络测试，程序运行结果并没有太大意义，但是至少表明 代码没有问题'''
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3
n = NeturalNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
n.query([1.0,0.5,1.5])

