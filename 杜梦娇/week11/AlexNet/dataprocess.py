import os
# 列出 ./data/image/train/ 目录下的所有文件，并将其存储在变量 photos 中
photos = os.listdir("./data/train/")

with open("data/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()
