import os

photos = os.listdir("./data/image/train/")

# 该部分用于将
''' 打开或创建一个名为 "data/dataset.txt" 的文件，用于写入处理后的图片信息。
    该文件将保存图片名称及其对应的类别标签，其中 "cat" 对应标签 0，"dog" 对应标签 1。 '''
with open("data/dataset.txt","w") as f:
    ''' 遍历图片列表 photos，处理每一张图片。 '''
    for photo in photos:
        ''' 提取图片文件名，去除文件扩展名。 '''
        name = photo.split(".")[0]
        ''' 判断图片名称，根据名称写入相应的类别标签。 '''
        if name=="cat":
            ''' 如果图片名称为 "cat"，则在文件中写入图片名称后跟 ";0"，表示猫的类别标签为 0。 '''
            f.write(photo + ";0\n")
        elif name=="dog":
            ''' 如果图片名称为 "dog"，则在文件中写入图片名称后跟 ";1"，表示狗的类别标签为 1。 '''
            f.write(photo + ";1\n")
''' 关闭文件，确保数据写入完成。 '''
f.close()
