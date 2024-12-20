from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
import AlexNet
import numpy as np
import cv2

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        x_train = []
        y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            label = lines[i].split(';')[1]
            img = cv2.imread("./data/train/" + name)
            img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            x_train.append(img)
            y_train.append(label)
            # 读完一个周期后重新开始
            i = (i + 1) % n 
        # 处理图像
        x_train = np.array(x_train)
        x_train = x_train.reshape(-1, 224, 224, 3)
        y_train = np_utils.to_categorical(np.array(y_train), num_classes=2)
        yield (x_train, y_train)

if __name__ == '__main__':
    # 模型保存的位置
    log_dir = "./logs/"
    # 打开图片列表 txt
    with open("./data/dataset.txt", "r") as f:
        lines = f.readlines()

    # 打乱行, 使数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%的数据用于训练, 10%用于推理
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet.AlexNet()

    # 保存的方式, 3epoch 保存一次
    checkpoint_period1 = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='accuracy', save_weights_only=False, save_best_only=True, period=3)
    # 学习率下降的方式, acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=1)
    # 是否需要早停, 当 val_loss 一直不下降就停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience= 10, verbose=1)

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ['accuracy']) # 0.001

    # 一次训练集大小
    batch_size = 128
    print('Train on {} samples, vals on {} samples, with batch size {}.'. \
          format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50, initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'last1.h5')