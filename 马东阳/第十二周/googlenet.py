
#   Inception-v1
'''
初始层
卷积层：GoogLeNet 开始于一个 7x7 的卷积层，通常带有步幅为 2，用于初步提取图像中的基本特征。
最大池化层：紧随其后的是一个 3x3 的最大池化层，步幅为 2，用于减小特征图的尺寸，降低计算量。
卷积层：之后是两个连续的 3x3 卷积层，用于进一步特征提取。
最大池化层：再次进行最大池化，继续减小特征图的尺寸。

Inception 模块
GoogLeNet 的核心是 Inception 模块，它们通常在初始层之后被多次重复。一个典型的 Inception 模块包含以下部分：
1x1 卷积：用于降维，减少后续计算的成本。
3x3 卷积：用于提取局部特征，通常在 1x1 卷积之后。
5x5 卷积：用于捕获更大范围的特征，同样在 1x1 卷积之后。
最大池化：用于捕捉图像中的重要特征，同时在池化之后使用 1x1 卷积以保持通道数。
所有这些操作的输出会被拼接在一起，形成 Inception 模块的输出。

辅助分类器
GoogLeNet 在中间部分包含一到两个辅助分类器，它们在训练期间提供额外的监督信号，帮助网络学习更加鲁棒的特征表示。
辅助分类器包括一个平均池化层，一个或多个 1x1 卷积层，一个全连接层，最后是一个 softmax 分类层。

输出层
在一系列 Inception 模块之后，GoogLeNet 使用全局平均池化层来将特征图转换为固定长度的向量。然后，这个向量
被馈送到一个全连接层，最后是 softmax 分类层，用于生成各个类别的概率预测。

具体结构
一个具体的 GoogLeNet 网络可能会包含多个 Inception 模块的堆叠，通常在模块之间会有最大池化层用于降低特征图
的尺寸。网络的总深度约为 22 层，这包括卷积层、Inception 模块和全连接层。

请注意，GoogLeNet 的具体实现可能因版本而异，比如 Inception-v2、Inception-v3、Inception-v4 等，它们
在 Inception 模块的设计和整个网络的结构上有所改进和调整。但是，上述描述概括了 GoogLeNet 的原始版本的基本
结构和设计思想。

'''




import torch.nn as nn
import torch
import torch.nn.functional as F

#   创建所需的模板文件
#   基本卷积模板
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#   Inception 结构模板
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        #   将输入特征矩阵分别输入到四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        #   将输出放入一个列表中
        outputs = [branch1, branch2, branch3, branch4]
        #   通过torch.cat合并四个输出，合并维度为1，即按照通道维度合并
        return torch.cat(outputs, 1)

#   InceptionAux 辅助分类器模板
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #   aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14

        x = self.averagePool(x)
        #   aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4

        x = self.conv(x)
        #   N x 128 x 4 x 4

        #   特征矩阵展平，从channel维度开始展平
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 2048

        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        #   N x 1024

        x = self.fc2(x)
        #   N x num_classes
        return x

#   定义GoogLeNet网络
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weight=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # ceil_mode=True 计算为小数时，向上取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        #   辅助分类器
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        #   AdaptiveAvgPool2d 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
# from model import GoogLeNet
import os
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

data_transform = {
    "train" : transforms.Compose([transforms.RandomResizedCrop(224),   # 随机裁剪
                                  transforms.RandomHorizontalFlip(), # 随机翻转
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]),
    "val" : transforms.Compose([transforms.Resize((224, 224)),    # 不能224，必须(224, 224)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])}

#   获取数据集所在的根目录
#   通过os.getcwd()获取当前的目录，并将当前目录与".."链接获取上一层目录
data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

#   获取数据集路径
image_path = data_root + "/data_set/"

#   加载数据集
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])

#   获取训练集图像数量
train_num = len(train_dataset)

#   获取分类的名称
#   {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list = train_dataset.class_to_idx

#   采用遍历方法，将分类名称的key与value反过来
cla_dict = dict((val, key) for key, val in flower_list.items())

#   将字典cla_dict编码为json格式
json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

#   定义模型
net = GoogLeNet(num_classes=5, aux_logits=True, init_weight=True)   # 实例化模型
net.to(device)
loss_function = nn.CrossEntropyLoss()   # 定义损失函数
#pata = list(net.parameters())   # 查看模型参数
optimizer = optim.Adam(net.parameters(), lr=0.0003)  # 定义优化器

#   设置存储权重路径
save_path = './googleNet.pth'
best_acc = 0.0
for epoch in range(1):
    # train
    net.train()  # 用来管理Dropout方法：训练时使用Dropout方法，验证时不使用Dropout方法
    running_loss = 0.0  # 用来累加训练中的损失
    for step, data in enumerate(train_loader, start=0):
        #   获取数据的图像和标签
        images, labels = data
        #   将历史损失梯度清零
        optimizer.zero_grad()

        #   参数更新
        #   因为采用了辅助分类器，得到了三个输出（主输出和两个辅助输出）
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        #   计算三个损失
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        #   将三个损失相加，得到最终损失
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()                                    # 误差反向传播
        optimizer.step()                                   # 更新节点参数

        #   打印统计信息
        running_loss += loss.item()
        #   打印训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()  # 关闭Dropout方法
    acc = 0.0
    #   验证过程中不计算损失梯度
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            #   acc用来累计验证集中预测正确的数量
            #   对比预测值与真实标签，sum()求出预测正确的累加值，item()获取累加值
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        #   如果当前准确率大于历史最优准确率
        if accurate_test > best_acc:
            #   更新历史最优准确率
            best_acc = accurate_test
            #   保存当前权重
            torch.save(net.state_dict(), save_path)
        #   打印相应信息
        print("[epoch %d] train_loss: %.3f  test_accuracy: %.3f"%
              (epoch + 1, running_loss / step, acc / val_num))

print("Finished Training")

import torch
# from model import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

#   加载预测图片
img = Image.open("./郁金香.png")
#   展示图片
plt.imshow(img)
#   图像预处理 [C, H, W]
img = data_transform(img)
#   扩充图像维度 [N, C, H, W]
img = torch.unsqueeze(img, dim=0)

# 读取 class_indict
try:
    json_file = open("./class_indices.json", "r")
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

#   初始化网络
model = GoogLeNet(num_classes=5, aux_logits=False)
#   加载权重
model_weight_path = "./googleNet.pth"
#   载入网络模型  strict=False不载入辅助分类器
missingkey, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
#   采用eval()模式，关闭Dropout方法
model.eval()
#   不去跟踪变量的损失梯度
with torch.no_grad():
    #   model(img)将图像输入模型得到输出，采用squeeze压缩维度，即将Batch维度压缩掉
    output = torch.squeeze(model(img))
    #   采用softmax将最终输出转化为概率分布
    predict = torch.softmax(output, dim=0)
    #   获取概率最大处的索引值
    predict_cla = torch.argmax(predict).numpy()

#   打印类别名称及其对应的预测概率
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
