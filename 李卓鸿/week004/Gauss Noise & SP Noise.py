import cv2
from numpy import shape
from skimage import util
import random
from matplotlib import pyplot

def Gauss_Noise(img,mu,sigma,per):
    re_img = img
    h,w = img.shape[:2]
    G_Num = int(h * w * per)
    for i in range(G_Num):
        random_x = random.randrange(0,h)
        random_y = random.randrange(0,w)

        pin = img[random_x,random_y]
        pout = pin + random.gauss(mu,sigma)

        if pout < 0 :
            pout = 0
        elif pout > 255:
            pout = 255

        re_img[random_x, random_y] = pout

    return re_img

def SP_Noise(img,snr):
    re_img = img
    h,w = img.shape[:2]
    np = int( snr * h * w )
    for i in range(np):
        random_x = random.randrange(0,h)
        random_y = random.randrange(0,w)

        re_img[random_x,random_y] = int( random.choice([0,255]) )

    return re_img

# 手动实现高斯、椒盐噪声
img = cv2.imread("lenna.png",0)
img1 = Gauss_Noise(img,8,8,0.8)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.imread("lenna.png",0)
img3 = SP_Noise(img,0.05)

cv2.imshow('Source img',img2)
cv2.imshow('Gauss Noise img',img1)
cv2.imshow('SP Noise img',img3)

cv2.waitKey(0)

# 调用噪声接口
img = cv2.imread('lenna.png')
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pyplot.subplot(221)
pyplot.title('source img')
pyplot.imshow(img_grey,cmap='gray')

pyplot.subplot(222)
pyplot.title('gauss noise img')
img = cv2.imread('lenna.png',0)
img_gauss = util.random_noise(img,mode='gaussian',mean=0.0315,var=0.0315)
pyplot.imshow(img_gauss,cmap='gray')

pyplot.subplot(223)
pyplot.title('sp noise img')
img = cv2.imread('lenna.png',0)
img_sp = util.random_noise(img,mode='s&p',amount=0.05)
pyplot.imshow(img_sp,cmap='gray')

pyplot.show()
