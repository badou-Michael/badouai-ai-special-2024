from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from alexnet import AlexNet
import numpy as np
import cv2
from keras import backend as K
import tensorflow as tf

#检查当前的图像数据格式是否为（N，C，H，W）
K.image_data_format()=='channels_first'

# 定义一个数据生成函数，用于从文件中读取数据并按批次生成训练数据
def generate_arrays_from_file(lines,batch_size):
    # 获取数据的总长度
    n=len(lines)
    # 初始化索引
    i=0
    # 无限循环，用于不断生成数据
    while 1:
        # 用于存储图像数据
        X_train=[]
        # 用于存储标签
        Y_train=[]
        for b in range(batch_size):
            if i==0:
                # np.random.shuffle原地打乱数组的顺序，用于提高训练的效率和泛化能力
                np.random.shuffle(lines)
            # 获取图像文件名，根据自定义的文件格式【cat.0.jpg;0】
            name = lines[i].split(';')[0]

            # 加载图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            # 转换图像颜色空间为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 图像归一化
            img = img / 255
            # 加入训练集
            X_train.append(img)
            # 加入标签
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始，更新索引，防止越界
            i = (i+1)%n
        # 处理图像，调整图像大小到（224，224）
        X_train = resize_image(X_train,(224,224))
        # 调整形状以适应keras输入 tf的图像数据格式为channels_last，数据形状为(batch_size, height, width, channels)
        X_train = X_train.reshape(-1,224,224,3)
        # 将整型标签转换为one-hot编码的形式‌。这种转换在分类任务中非常常用，特别是在处理多分类问题时，将类别标签转换为one-hot编码矩阵可以方便地进行模型训练和评估
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes=2)
        # 返回图像数据和标签数据
        yield (X_train,Y_train)


# 定义图像大小调整函数
def resize_image(image, size):
    # 使用tensorFlow命名空间以组织操作
    with tf.name_scope('resize_image'):
        # 用于存储调整大小后的图像
        images = []
        # 遍历图像列表
        for i in image:
            # 调整图像大小
            i = cv2.resize(i, size)
            # 加入调整后的图像
            images.append(i)
            # 转为Numpy数组
        images = np.array(images)
        #返回调整大小后的图像数组
        return images

if __name__ == "__main__":
    log_dir="./logs/"
    with open(r".\data\dataset.txt", "r") as f:
        # 按行读取内容，每行通常是文件名和标签的组合
        lines = f.readlines()
    np.random.seed(0)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 按比例划分训练集和验证集（90%训练，10%验证）
    num_val = int(len(lines) * 0.1)  # 验证集数量
    num_train = len(lines) - num_val  # 训练集数量

    # 创建AlexNet模型实例
    model = AlexNet()

    # 设置模型保存回调函数
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',  # 保存文件名格式
        monitor='acc',  # 监控指标为训练精度
        save_weights_only=False,  # 保存完整模型
        save_best_only=True,  # 只保存性能最优的模型
        period=3  # 每3个epoch保存一次
    )

    # 设置学习率调整回调函数
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',  # 监控指标为训练精度
        factor=0.5,  # 学习率每次减少为原来的一半
        patience=3,  # 如果精度连续3次无提升，则调整学习率
        verbose=1  # 输出调整信息
    )

    # 设置早停回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        min_delta=0,  # 损失的最小变化值
        patience=10,  # 如果连续10次无改善，则停止训练
        verbose=1  # 输出信息
    )

    # 编译模型
    model.compile(
        loss='categorical_crossentropy',  # 使用交叉熵作为损失函数
        optimizer=Adam(lr=1e-3),  # 使用Adam优化器，学习率为0.001
        metrics=['accuracy']  # 监控准确率
    )

    # 定义批量大小,即一次的训练集大小
    batch_size = 128

    # 输出训练和验证数据的统计信息
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练模型
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),  # 训练数据生成器
        steps_per_epoch=max(1, num_train // batch_size),  # 每epoch的训练步数
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),  # 验证数据生成器
        validation_steps=max(1, num_val // batch_size),  # 每epoch的验证步数
        epochs=50,  # 最大训练50个epoch
        initial_epoch=0,  # 从第0个epoch开始
        callbacks=[checkpoint_period1, reduce_lr]  # 使用回调函数
    )

    # 保存训练结束后的模型权重
    model.save_weights(log_dir + 'last1.h5')
