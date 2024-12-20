import os

photos = os.listdir("./data/train/")
print(photos)
with open("data/dataset.txt","w") as f:
    print(photos[0])
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()
