import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
from keras import backend as K

# 设置Keras图像数据格式
K.image_data_format() == "channels_first"


class ImageDataGenerator:
    """图像数据生成器类"""

    def __init__(self, data_lines, batch_size, image_size=(224, 224)):
        self.data_lines = data_lines
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_length = len(data_lines)
        self.current_index = 0

    def read_image(self, image_name):
        """读取并预处理单张图片"""
        # 读取图片
        image_path = r".\data\image\train" + "/" + image_name
        image = cv2.imread(image_path)
        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 归一化
        image = image / 255.0
        return image

    def get_batch_data(self):
        """获取一批数据"""
        images = []
        labels = []

        for _ in range(self.batch_size):
            # 当读完所有数据时，重新打乱
            if self.current_index == 0:
                np.random.shuffle(self.data_lines)

            # 获取当前行数据
            current_line = self.data_lines[self.current_index]
            image_name = current_line.split(";")[0]
            label = current_line.split(";")[1]

            # 读取图片
            image = self.read_image(image_name)

            images.append(image)
            labels.append(label)

            # 更新索引
            self.current_index = (self.current_index + 1) % self.data_length

        # 处理图片尺寸
        images = np.array(images)
        images = cv2.resize(images, self.image_size)
        images = images.reshape(-1, self.image_size[0], self.image_size[1], 3)

        # 处理标签
        labels = np_utils.to_categorical(np.array(labels), num_classes=2)

        return images, labels

    def generate(self):
        """生成器函数"""
        while True:
            yield self.get_batch_data()


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, model, batch_size=128):
        self.model = model
        self.batch_size = batch_size
        self.log_dir = "./logs/"

    def setup_callbacks(self):
        """设置回调函数"""
        callbacks = []

        # 模型检查点
        checkpoint = ModelCheckpoint(
            self.log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
            monitor="acc",
            save_weights_only=False,
            save_best_only=True,
            period=3,
        )
        callbacks.append(checkpoint)

        # 学习率调整
        reduce_lr = ReduceLROnPlateau(monitor="acc", factor=0.5, patience=3, verbose=1)
        callbacks.append(reduce_lr)

        # 早停
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, verbose=1
        )
        callbacks.append(early_stopping)

        return callbacks

    def train(self, train_data, val_data, epochs=50):
        """训练模型"""
        # 编译模型
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=1e-3),
            metrics=["accuracy"],
        )

        # 计算步数
        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        validation_steps = max(1, len(val_data) // self.batch_size)

        # 创建数据生成器
        train_generator = ImageDataGenerator(train_data, self.batch_size)
        val_generator = ImageDataGenerator(val_data, self.batch_size)

        # 打印训练信息
        print(
            f"训练样本数: {len(train_data)}, 验证样本数: {len(val_data)}, "
            f"批次大小: {self.batch_size}"
        )

        # 开始训练
        self.model.fit_generator(
            train_generator.generate(),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator.generate(),
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=self.setup_callbacks(),
        )

        # 保存最终权重
        self.model.save_weights(self.log_dir + "last1.h5")


def main():
    """主函数"""
    # 读取数据集
    with open(r".\data\dataset.txt", "r") as f:
        data_lines = f.readlines()

    # 打乱数据
    np.random.seed(10101)
    np.random.shuffle(data_lines)
    np.random.seed(None)

    # 划分训练集和验证集
    val_split = 0.1
    val_size = int(len(data_lines) * val_split)
    train_data = data_lines[:-val_size]
    val_data = data_lines[-val_size:]

    # 创建模型
    model = AlexNet()

    # 创建训练器并开始训练
    trainer = ModelTrainer(model)
    trainer.train(train_data, val_data)


if __name__ == "__main__":
    main()
