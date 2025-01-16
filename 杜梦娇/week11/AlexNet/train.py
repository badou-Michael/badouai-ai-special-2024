
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
#检查图像数据的维度顺序是 (c, h, w)
K.image_data_format() == 'channels_first'

# 定义读取数据函数
def generate_arrays_from_file(lines,batch_size):
    # 获取列表总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)# 转为灰度图像
            img = img/255 # 归一化
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        # 重塑图像数据为模型输入数据
        X_train = X_train.reshape(-1,224,224,3)
        # 标签one-hot化
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)
        # 生成器函数产生一个包含处理后的图像和标签的元组
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()
    # 打乱数据
    np.random.seed(10101)#设置随机数生成器种子
    np.random.shuffle(lines)
    np.random.seed(None)#随机数生成器将恢复到其默认行为

    # 数据集划分：训练集和测试集 90%作为训练集
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    #初始化模型
    model = AlexNet()

    # 在训练过程中定期保存模型的状态设置，每5个epoch保存一次，只保存模型权重，且只保存模型表现最好的参数
    checkpoint_period1 = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='acc', save_weights_only=False, save_best_only=True, period=5)

    # 设置学习率下降，ACC连续三次没有改善，则下降学习率，学习率衰减因子为0.5，即变为原来的一半
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)

    # 设置earlystopping, 当loss连续十个epoch不下降时则停止模型训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 设置交叉熵损失函数、优化器（学习率为0.001）、模型评估指标ACC
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    #设置batch大小
    batch_size = 128

    # 训练模型
    model.fit_generator(generate_arrays_from_file(lines[:, num_train], batch_size), steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=generate_arrays_from_file((lines[num_train,:], batch_size), validation_steps=max(1, num_val//batch_size)),
                        epochs=100, initial_epoch=0, callbacks=[checkpoint_period1, reduce_lr, early_stopping])

    # 保存训练好的模型
    model.save_weights(log_dir+'AlexNetLastModel.h5')
