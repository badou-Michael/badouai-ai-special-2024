"""
利用函数接口实现高斯噪声和椒盐噪声
"""
import cv2
# import skimage as sk
from skimage import util
img1 = cv2.imread("..\\lenna.png")
img2 = cv2.imread("..\\lenna.png",0)
# 高斯噪声
noise_gs_img = util.random_noise(img1,mode='gaussian',mean=0.4 ,var=0.03) ##var --> 方差
cv2.imshow("gauss img:",noise_gs_img)

# 椒盐噪声
salt_pepper_noise = util.random_noise(img2,mode='s&p',amount=0.3,salt_vs_pepper=10)  ##amount -->信噪比;slat_vs_pepper -->盐噪声与椒噪声比例
cv2.imshow("salt perpper imag",salt_pepper_noise)

cv2.waitKey()
