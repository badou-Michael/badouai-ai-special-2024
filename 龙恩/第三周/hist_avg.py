import numpy as np
import cv2
from matplotlib import pyplot as plt

def ploting(place, name,img):
    plt.subplot(place)
    plt.title(name)
    plt.plot(img)

def equ(img):
    img=cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst=cv2.equalizeHist(gray)

    hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([dst],[0],None,[256],[0,256])

    ploting(121,"gray",hist1)
    ploting(122,"original",hist2)
    plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey(0)

    (b, g, r) = cv2.split(img)
    H_b = cv2.equalizeHist(b)
    H_g = cv2.equalizeHist(g)
    H_r = cv2.equalizeHist(r)
    result = cv2.merge((H_b, H_g, H_r))
    cv2.imshow("Histogram Equalization", np.hstack([img, result]))
    plt.show()

equ("lenna.png")
