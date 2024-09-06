from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def ploting(name,place,title):
    plt.subplot(place)
    fig = cv2.cvtColor(name, cv2.COLOR_BGR2RGB)
    if len(name.shape) == 2 or name.shape[2] == 1:  # Grayscale image
        plt.imshow(name,cmap='gray')
    else:  # Color image
        fig = cv2.cvtColor(name, cv2.COLOR_BGR2RGB)
        plt.imshow(fig)
    plt.title(title)
    plt.axis('off')

def newimage(source):
    #Original png
    img=cv2.imread(source)
    ploting(img,221,"Original Image")

    #Gray scale png
    height,width=img.shape[:2]
    img_gray=np.zeros((height,width),img.dtype)
    for i in range(height):
        for j in range(width):
            img_gray[i][j]=int(img[i][j][0]*0.11+img[i][j][1]*0.59+img[i][j][2]*0.3)
    ploting(img_gray,222,"Gray Scale Image")

    #Black and White Image
    img_dark=np.zeros((height,width),img.dtype)
    for i in range(height):
        for j in range(width):
            img_dark[i][j]=0 if img_gray[i][j]<255/2 else 255
    ploting(img_dark,223,"Binary Image")

    #plt.savefig(f"New_{source}")
    plt.show()


newimage("lenna.png")


'''
Using
img=plt.imread(source)
plt.imshow(img)
///cv2.imshow("image show gray",img_gray)
OR
img=cv2.imread(source)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

'''
