from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

K.image_data_format() == 'channels_first'  # 通道读取的默认顺序是HWC，但这个语句将顺序指明为CHW


def generate_arrays_from_file(lines, batch_size):   # lines:文件数据行列表、batch_size:每个批次中的样本数
    # generate_arrays_from_file函数是一个生成器，当数据量大导致无法一次性加载到内存中时，就需要使用生成器按需生成数据批次
    # 获取总长度
    n = len(lines)
    i = 0   # 用于迭代数据行的索引
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:  # 每遇见一个新批次就随机打乱lines列表
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]   # 分割 lines[i] 来获取图像名称和标签
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255     # 对像素值归一化
            # 将图像添加到X_train列表中，将标签添加到Y_train列表中
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))   # 调用utils文件中的resize_image函数将图像调整到224x224的大小
        X_train = X_train.reshape(-1, 224, 224, 3)  # 将 X_train 重塑为四维数组，以符合 Keras 模型的输入要求（样本数batch_size，高度，宽度，通道数）
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        # np_utils.to_categorical 将标签转换为 one-hot 编码
        yield (X_train, Y_train)    # 使用 yield 关键字生成一个包含 (X_train, Y_train) 的批次


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    '''
    1.with语句用于包装执行代码块，并确保文件在使用后正确关闭
    2.r".\data\dataset.txt"是文件的路径，前面的r表示原始字符串，这样写可以避免在字符串中使用反斜杠（\）时的转义问题
    3."r"表示以只读模式打开文件。
    '''
    with open(r".\data\dataset.txt", "r") as f:  # 将打开的文件对象赋值给变量f
        lines = f.readlines()  # readlines()——读取文件中的所有行，并将它们作为字符串列表返回，其中每个字符串都是文件中的一行、包括行尾的换行符

    # 按行打乱数据
    np.random.seed(10101)
    '''
    设置NumPy随机数生成器的种子为10101，设置随机数种子是为了确保代码的可重复性，使得在不同的时间或环境中都能重现结果
    使用相同的种子时，无论运行代码多少次，np.random函数产生的随机数序列都是相同的
    '''
    np.random.shuffle(lines)  # 打乱字符串列表lines中的元素顺序
    np.random.seed(None)  # 重置随机数生成器

    # 划分训练集和测试集的样本总数——90%用于训练、10%用于估计
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()  # 创建Alexnet模型的实例化对象model

    # 每3个epoch检查一次是否要保存模型
    checkpoint_period1 = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                         monitor='acc', save_weights_only=False, save_best_only=True, period=3)
    '''
    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5':这是模型保存的文件名格式和路径
    monitor='acc': 指定要监控的量是准确率（accuracy）
    save_weights_only=False: 表示保存整个模型，而不仅仅是模型的权重
    save_best_only=True: 表示只保存在验证集上表现最好的模型
    period=3: 表示每3个epoch检查一次是否要保存模型
    '''

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    # 当准确率acc在指定的patience个epoch内没有提升时，将调整学习率
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
    '''
    monitor='acc': 同样，这里监控的是准确率
    factor=0.5: 学习率调整的因子，即每次调整时将当前学习率乘以0.5
    patience=3: 如果在3个epoch内监控的量没有改进，则调整学习率
    verbose=1: 打印详细信息
    '''

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    '''
    monitor='val_loss': 监控验证集上的损失
    min_delta=0: 如果监控的量在patience个epoch内没有至少减少min_delta，则停止训练
    patience=10: 如果在10个epoch内验证损失没有改进，则停止训练
    '''

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    '''
    loss='categorical_crossentropy': 使用分类交叉熵作为损失函数，这通常用于多分类问题
    optimizer=Adam(lr=1e-3): 使用Adam优化器，并设置初始学习率为0.001
    metrics=['accuracy']: 在训练和测试时评估模型的准确率
    '''

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir + 'last1.h5')    # 训练结束后保存模型的权重到指定文件，这不会保存模型的架构、只保存模型权重
