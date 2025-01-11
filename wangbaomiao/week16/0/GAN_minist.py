# -*- coding: utf-8 -*-
# time: 2024/12/2 16:00
# file: GAN_minist.py
# author: flame
import os

import numpy as np
from keras import Sequential, Model
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# 设置环境变量，减少TensorFlow的日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

''' 定义生成对抗网络（GAN）类，包含生成器和判别器的构建、训练和图像采样方法。 '''
class GAN():
    ''' 初始化方法，设置图像属性和模型参数。 '''
    def __init__(self):
        ''' 图像的行数，MNIST数据集的图像高度为28像素。 '''
        self.img_rows = 28
        ''' 图像的列数，MNIST数据集的图像宽度为28像素。 '''
        self.img_cols = 28
        ''' 图像的通道数，MNIST数据集为灰度图像，通道数为1。 '''
        self.channels = 1
        ''' 图像的形状，表示为 (高度, 宽度, 通道数)。 '''
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        ''' 隐含层的维度，用于生成器的输入。 '''
        self.latent_dim = 100

        ''' 创建优化器，使用Adam优化器，学习率为0.0002，β1为0.5。 '''
        optimizer = Adam(0.0002, 0.5)
        ''' 构建并编译判别器模型。 '''
        self.discriminator = self.build_discriminator()
        ''' 编译判别器模型，损失函数为二元交叉熵，优化器为Adam，评估指标为准确率。 '''
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        ''' 构建生成器模型。 '''
        self.generator = self.build_generator()
        ''' 创建隐含层输入张量，形状为 (latent_dim,)。 '''
        z = Input(shape=(self.latent_dim,))
        ''' 生成器将隐含层输入转换为图像。 '''
        img = self.generator(z)
        ''' 判别器对生成的图像进行评估。 '''
        validity = self.discriminator(img)
        ''' 在训练生成器时，冻结判别器的权重。 '''
        self.discriminator.trainable = False
        ''' 构建组合模型，输入为隐含层，输出为判别器对生成图像的评估结果。 '''
        self.combined = Model(z, validity)
        ''' 编译组合模型，损失函数为二元交叉熵，优化器为Adam。 '''
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    ''' 构建判别器模型，用于区分真实图像和生成图像。 '''
    def build_discriminator(self):
        ''' 创建顺序模型。 '''
        model = Sequential()
        ''' 将输入图像展平为一维向量。 '''
        model.add(Flatten(input_shape=self.img_shape))
        ''' 添加全连接层，输出维度为512。 '''
        model.add(Dense(512))
        ''' 添加LeakyReLU激活函数，α为0.2。 '''
        model.add(LeakyReLU(alpha=0.2))
        ''' 添加全连接层，输出维度为256。 '''
        model.add(Dense(256))
        ''' 添加LeakyReLU激活函数，α为0.2。 '''
        model.add(LeakyReLU(alpha=0.2))
        ''' 添加全连接层，输出维度为1，激活函数为sigmoid。 '''
        model.add(Dense(1, activation='sigmoid'))
        ''' 打印模型结构摘要。 '''
        model.summary()
        ''' 创建输入张量，形状为 (img_shape,)。 '''
        img = Input(shape=self.img_shape)
        ''' 判别器对输入图像进行评估。 '''
        validity = model(img)
        ''' 返回判别器模型。 '''
        return Model(img, validity)

    ''' 构建生成器模型，用于生成图像。 '''
    def build_generator(self):
        ''' 创建顺序模型。 '''
        model = Sequential()
        ''' 添加全连接层，输入维度为latent_dim，输出维度为256。 '''
        model.add(Dense(256, input_dim=self.latent_dim))
        ''' 添加LeakyReLU激活函数，α为0.2。 '''
        model.add(LeakyReLU(alpha=0.2))
        ''' 添加批量归一化层，动量为0.8。 '''
        model.add(BatchNormalization(momentum=0.8))
        ''' 添加全连接层，输出维度为512。 '''
        model.add(Dense(512))
        ''' 添加LeakyReLU激活函数，α为0.2。 '''
        model.add(LeakyReLU(alpha=0.2))
        ''' 添加批量归一化层，动量为0.8。 '''
        model.add(BatchNormalization(momentum=0.8))
        ''' 添加全连接层，输出维度为1024。 '''
        model.add(Dense(1024))
        ''' 添加LeakyReLU激活函数，α为0.2。 '''
        model.add(LeakyReLU(alpha=0.2))
        ''' 添加批量归一化层，动量为0.8。 '''
        model.add(BatchNormalization(momentum=0.8))
        ''' 添加全连接层，输出维度为图像的总像素数，激活函数为tanh。 '''
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        ''' 将输出重塑为图像形状。 '''
        model.add(Reshape(self.img_shape))
        ''' 打印模型结构摘要。 '''
        model.summary()
        ''' 创建输入张量，形状为 (latent_dim,)。 '''
        model_input = Input(shape=(self.latent_dim,))
        ''' 生成器将输入张量转换为图像。 '''
        img = model(model_input)
        ''' 返回生成器模型。 '''
        return Model(model_input, img)

    ''' 训练GAN模型。 '''
    def train(self, epochs, batch_size=128, sample_interval=50):
        ''' 加载MNIST数据集，只使用训练集的图像。 '''
        (X_train, _), (_, _) = mnist.load_data()
        ''' 将图像像素值缩放到-1到1之间。 '''
        X_train = X_train / 127.5 - 1
        ''' 增加一个通道维度。 '''
        X_train = np.expand_dims(X_train, axis=3)
        ''' 创建真实样本标签，值为1。 '''
        real = np.ones((batch_size, 1))
        ''' 创建虚假样本标签，值为0。 '''
        fake = np.zeros((batch_size, 1))
        ''' 开始训练循环。 '''
        for epoch in range(epochs):
            ''' 随机选择一批图像索引。 '''
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            ''' 获取选定的图像。 '''
            imgs = X_train[idx]
            ''' 从标准正态分布中生成隐含层输入。 '''
            z = np.random.normal(0, 1, (batch_size, self.latent_dim))
            ''' 生成器生成一批虚假图像。 '''
            gen_imgs = self.generator.predict(z)
            ''' 训练判别器，输入真实图像和真实标签。 '''
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            ''' 训练判别器，输入虚假图像和虚假标签。 '''
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            ''' 计算判别器的平均损失。 '''
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            ''' 从标准正态分布中生成新的隐含层输入。 '''
            z = np.random.normal(0, 1, (batch_size, self.latent_dim))
            ''' 训练生成器，目标是使判别器认为生成的图像是真实的。 '''
            g_loss = self.combined.train_on_batch(z, real)
            ''' 打印当前训练进度。 '''
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            ''' 每隔sample_interval个epoch保存生成的图像。 '''
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    ''' 生成并保存图像样本。 '''
    def sample_images(self, epoch):
        ''' 设置生成图像的数量为3x3。 '''
        r, c = 3, 3
        ''' 从标准正态分布中生成隐含层输入。 '''
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        ''' 生成器生成一批图像。 '''
        gen_imgs = self.generator.predict(noise)
        ''' 将生成的图像从-1到1的范围缩放到0到1之间。 '''
        gen_imgs = 0.5 * gen_imgs + 0.5
        ''' 创建画布和子图。 '''
        fig, axs = plt.subplots(r, c)
        ''' 用于遍历生成的图像。 '''
        count = 0
        ''' 遍历生成的图像并绘制。 '''
        for i in range(r):
            for j in range(c):
                ''' 绘制生成的图像。 '''
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                ''' 关闭坐标轴。 '''
                axs[i, j].axis('off')
                ''' 更新计数器。 '''
                count += 1
        ''' 保存生成的图像。 '''
        fig.savefig("images/mnist_%d.png" % epoch)
        ''' 关闭图像。 '''
        plt.close()

''' 主程序入口，创建GAN实例并开始训练。 '''
if __name__ == '__main__':
    ''' 创建GAN实例。 '''
    gan = GAN()
    ''' 开始训练，设置训练轮数为3000，批次大小为32，每隔200个epoch保存一次图像。 '''
    gan.train(epochs=3000, batch_size=32, sample_interval=200)
