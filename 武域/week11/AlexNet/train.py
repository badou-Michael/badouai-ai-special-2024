from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.optimizers import Adam
from model import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

K.set_image_data_format('channels_first')


def generate_arr_from_file(lines, batch_size):
    l = len(lines)
    i = 0
    while 1:
        x_train = []
        y_train = []
        for batch in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            tag = lines[i].split(';')[1]
            img = cv2.imread(r'data/image/train'+ '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            x_train.append(img)
            y_train.append(tag)
            i = (i + 1) % l

        x_train = utils.resized_image(x_train, (224, 224))
        x_train = x_train.reshape((-1, 224, 224, 3))
        y_train = np_utils.to_categorical(np.array(y_train), num_classes=2)
        yield x_train, y_train


if __name__ == '__main__':
    log_dir = 'log'
    with open (r"data/dataset.txt", r) as f:
        lines = f.readlines()

        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

        num_val = int(len(lines) * 0.1)
        num_train = len(lines) - num_val

        model = AlexNet()

        # save best result per 3 period
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='acc',
                                     save_best_only=True,
                                     save_weights_only=False,
                                     period=3)

        # reduce lr if acc didn't increase for 3 times
        reduce_lr = ReduceLROnPlateau(monitor='acc',
                                      factor=0.5,
                                      patience=3,
                                      verbose=1,)

        # early stopping if val_loss didn't decrease for 10 epochs
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arr_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arr_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')



