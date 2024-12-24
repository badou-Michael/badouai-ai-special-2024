from keras.optimizers import Adam
from lesson11_2_AlexNet_CNNbuild import AlexNet
from lesson11_2_Preprocess_data import data_set
from lesson11_2_Generator import generator
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = r'alexnet/AlexNet-Keras-master/logs/'
model = AlexNet().model

# 定义batch大小
batch_size = 128

# 读取dataset.txt文件
with open(r'alexnet\AlexNet-Keras-master\data\dataset.txt', 'r') as f:
    pipeline = f.readlines()
np.random.shuffle(pipeline)

# 设置训练集和验证集
num_train, num_val = data_set()

# 设定一个自动存档点，当执行每3次epoch后，保存一次权重，并只保留acc最高的那次
checkpoint = ModelCheckpoint(
    filepath=model_path + 'epoch：{epoch}-loss：{loss:.3f}-val_loss：{val_loss:.3f}.h5',
    monitor='accuracy',
    save_best_only=True,
    save_weights_only=False,
    period=3,
)

# 设置一个自动化降低学习率的函数，监测loss的下降速度，3轮下降没超过0.001，降低20%学习率
reduce_lr = ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.8,
    patience=3,
    verbose=1,
    min_delta=0.001,
)

# 设置一个自动化停止训练的函数，监测val_loss的下降速度，如果10次一点没下降则停止验证
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
)

# 为模型定义好一些参数细节
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

# 训练
model.fit(
    generator(pipeline[:num_train], batch_size),
    steps_per_epoch=max(1, num_train // batch_size),
    validation_data=generator(pipeline[num_train:], batch_size),
    validation_steps=max(1, num_val // batch_size),
    epochs=50,
    verbose=1,
    callbacks=[checkpoint, reduce_lr, early_stopping],
)

model.save_weights(os.path.join(model_path, 'last1.h5'))




