from keras.applications import vgg16
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import utils

tf.compat.v1.enable_eager_execution()

import Cifar10_data

# 用Cifar10的数据训练alexnet，以及vgg16的训练

# 常值定义
max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="./Cifar_data/cifar-10-batches-bin"

# 加载数据
train_images,train_labels=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
test_images,test_labels=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)


def generate_arrays_from_data(images, labels, batch_size):
    while True:  # 保证生成器可以无限生成数据
        num_samples = images.shape[0]

        # 确保每个批次的大小不会超出数据集的总大小
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # 进行必要的归一化（CIFAR-10 图像的像素值在 0 到 255 之间）
            batch_images = batch_images / 255.0  # 归一化到 [0, 1]

            # 转为 NumPy 数组
            if isinstance(batch_images, tf.Tensor):
                batch_images = batch_images.numpy()
            if isinstance(batch_labels, tf.Tensor):
                batch_labels = batch_labels.numpy()

            # 打印批次的形状以便调试
            print(f"Batch images shape: {batch_images.shape}, Batch labels shape: {batch_labels.shape}")

            yield batch_images, batch_labels

#定义AlexNet模型
def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # AlexNet
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 两个全连接层，最后输出为1000类,这里改为2类（猫和狗）
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model

if __name__ == "__main__":
# AlexNet模型训练
    # 模型保存的位置
    log_dir = "./logs/"

    # 建立AlexNet模型
    model = AlexNet()

    # 设置保存方式
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 设置学习率
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 设置早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])


    # 开始训练
    model.fit_generator(
        generate_arrays_from_data(train_images, train_labels, batch_size),
        steps_per_epoch=max(1, train_images.shape[0] // batch_size),
        validation_data=generate_arrays_from_data(test_images, test_labels, batch_size),
        validation_steps=max(1, test_images.shape[0] // batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]
    )
    model.save_weights(log_dir + 'last1.h5')

# vgg16模型训练
    # 读取并处理输入图片
    img_path = "./test_data/table.jpg"
    img = utils.load_image(img_path)
    resized_img = utils.resize_image(img, (224, 224))

    # 创建占位符和模型
    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # 输入大小固定为224x224
    prediction = vgg16.vgg_16(inputs)  # 使用VGG16模型进行预测

    # 创建TensorFlow会话并加载预训练模型
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt_filename = './model/vgg_16.ckpt'
        saver.restore(sess, ckpt_filename)  # 恢复模型

        # 执行预测并打印结果
        softmax_probs = tf.nn.softmax(prediction)
        prob_values = sess.run(softmax_probs, feed_dict={inputs: resized_img})

        # 输出预测结果
        print("Prediction Result:")
        utils.print_prob(prob_values[0], './synset.txt')
