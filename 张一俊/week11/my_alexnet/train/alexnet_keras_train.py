# 用 AlexNet 架构训练一个图像分类模型

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from my_alexnet.model.alexnet import AlexNet
import numpy as np
from my_alexnet.train import utils
import cv2

K.image_data_format() == 'channels_first'

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)  # 总数据量
    i = 0
    while 1:  # 无限循环生成数据
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)  # 在每个epoch开始时打乱数据
            name = lines[i].split(';')[0]
            img = cv2.imread(r".\train_data" + '/' + name)  # 读取图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            img = img / 255  # 归一化
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])  # 获取标签（假设是分类标签）
            i = (i + 1) % n  # 循环读取数据

        X_train = utils.resize_image(X_train, (227, 227))  # 调整图像大小
        X_train = X_train.reshape(-1, 227, 227, 3)  # 适应输入层格式
        Y_train = to_categorical(np.array(Y_train), num_classes=2)  # 标签的One-hot编码
        yield (X_train, Y_train)  # 生成器返回一批数据


if __name__ == "__main__":
    log_dir = "./logs/"  # 存放模型日志和检查点的路径

    # 打开并读取数据集
    with open(r".\dataset.txt", "r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)  # 打乱数据集
    np.random.seed(None)

    # 划分训练集和验证集
    num_val = int(len(lines) * 0.1)  # 10%作为验证集
    num_train = len(lines) - num_val  # 其余作为训练集

    # 创建AlexNet模型
    model = AlexNet()

    # 设置模型检查点（保存最好的模型）
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    # 设置学习率下降策略
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 设置早停策略
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    # 训练的batch大小
    batch_size = 128

    print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr, early_stopping])

    model.save_weights(log_dir + 'last1.h5')  # 保存训练结束后的模型权重
