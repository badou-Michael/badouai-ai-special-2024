import cv2
import tensorflow.keras.losses
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from week11.AlexNet.model.AlexNet import AlexNet, AlexNet2
import util
import h5py
from tensorflow.keras.utils import to_categorical

# K.set_image_data_format('channels_first')
print(K.image_data_format() == 'channels_first')

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread("./data/train/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            i = (i + 1) % n
        X_train = util.resize_images(X_train, (224, 224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = to_categorical(np.array(Y_train), num_classes = 2)
        yield X_train, Y_train

if __name__ == "__main__":
    log_dir = "./log"
    # 读取数据集
    with open(r"./data/dataset.txt", "r") as f:
        lines = f.readlines()
    # 打乱行
    np.random.seed(3209)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 测试、训练集数量
    num_eval = int(len(lines) * 0.1)
    num_train = len(lines) - num_eval

    model = AlexNet()
    # 指定保存方式
    checkpoint_period = ModelCheckpoint(
        log_dir + "epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor="acc",
        save_best_only=True,
        period=3
    )
    # 学习率下降方式
    reduce_lr = ReduceLROnPlateau(
        monitor="acc",
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 提前终止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    # 交叉熵
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_eval, batch_size))

    model.fit_generator(
        generator=generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=min(1, num_train / batch_size),
        epochs=3,
        verbose=1,
        callbacks=[checkpoint_period, reduce_lr, early_stopping],
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=min(1, num_eval / batch_size)
    )

    model.save_weights(log_dir + "/last1.h5")



