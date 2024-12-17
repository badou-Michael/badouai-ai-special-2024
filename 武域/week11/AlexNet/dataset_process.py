import os

photos = os.listdir('data/image/train')

with open('data/dataset.txt', 'w') as f:
    for photo in photos:
        label = photo.split('.')[0]
        if label == 'cat':
            f.write(photo + ";0\n")
        elif label == 'dog':
            f.write(photo + ";1\n")
        else:
            print(photo + " is not a cat or dog.")
f.close()
