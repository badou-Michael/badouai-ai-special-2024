from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

# 设置图像数据格式（根据当前 Keras 版本，默认 'channels_last' 或 'channels_first'）
K.image_data_format() == 'channels_first'  # 当前脚本中没有使用此变量，可能需要注释掉


def generate_data_from_file(file_lines, batch_size):
    """
    从文件中读取图像和标签，生成批次数据。

    参数：
    - file_lines: 数据集文件路径（每一行包含图像路径和标签）
    - batch_size: 每个批次的样本数量

    返回：
    - 图像数据和标签的批次生成器
    """
    total_samples = len(file_lines)
    index = 0  # 用于遍历数据

    while True:
        X_batch = []
        Y_batch = []

        # 获取一个批次数据
        for batch_index in range(batch_size):
            if index == 0:  # 每个周期开始时打乱数据
                np.random.shuffle(file_lines)

            # 获取文件名和标签
            file_name = file_lines[index].split(';')[0]
            label = file_lines[index].split(';')[1]

            # 读取图像
            img = cv2.imread(r".\data\image\train" + '/' + file_name)
            print(f"Image Shape: {img.shape}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
            img = img / 255.0  # 归一化到0~1之间

            X_batch.append(img)
            Y_batch.append(label)

            # 更新索引，循环读取数据
            index = (index + 1) % total_samples

        # 对图像进行尺寸调整
        X_batch = utils.resize_image(X_batch, (224, 224))  # 调整图像为224x224
        X_batch = np.array(X_batch).reshape(-1, 224, 224, 3)  # 确保数据格式为4D（batch_size, height, width, channels）

        # 转换标签为one-hot编码
        Y_batch = np_utils.to_categorical(np.array(Y_batch), num_classes=2)  # 假设有2个类别

        yield X_batch, Y_batch  # 返回一个批次的数据和标签


if __name__ == "__main__":
    # 模型保存的路径
    logs_directory = "./logs/"

    # 读取数据集文件
    with open(r".\data\dataset.txt", "r") as f:
        dataset_lines = f.readlines()

    # 打乱数据
    np.random.seed(10101)
    np.random.shuffle(dataset_lines)
    np.random.seed(None)

    # 90%数据用于训练，10%数据用于验证
    validation_samples = int(len(dataset_lines) * 0.1)
    training_samples = len(dataset_lines) - validation_samples

    # 构建AlexNet模型
    model = AlexNet()

    # 模型保存方式：每3个epoch保存一次最好的模型
    checkpoint_callback = ModelCheckpoint(
        logs_directory + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='accuracy',
        save_weights_only=False,
        save_best_only=True,
        period=3  # 每3个epoch保存一次
    )

    # 学习率调整：当accuracy停滞时，减半学习率
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 早停策略：当val_loss不再下降时，停止训练
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 编译模型，使用交叉熵损失和Adam优化器
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    # 每个批次的训练样本数量
    batch_size = 128

    print(
        f"Train on {training_samples} samples, validate on {validation_samples} samples, with batch size {batch_size}.")

    # 开始训练模型
    model.fit_generator(
        generate_data_from_file(dataset_lines[:training_samples], batch_size),
        steps_per_epoch=max(1, training_samples // batch_size),
        validation_data=generate_data_from_file(dataset_lines[training_samples:], batch_size),
        validation_steps=max(1, validation_samples // batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_callback, reduce_lr_callback, early_stopping_callback]  # 添加回调函数
    )

    # 保存最后的权重
    model.save_weights(logs_directory + 'last_weights.h5')
