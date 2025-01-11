import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util
#实现噪声接口的调用
photo=cv.imread("ho.jpg")
#高斯分布的加性噪声
noise_lr_photo=util.random_noise(photo,mode='localvar')
cv.imshow("localvar",noise_lr_photo)
#泊松噪声
noise_pn_photo=util.random_noise(photo,mode='poisson')
cv.imshow("poisson",noise_pn_photo)
#盐噪声
noise_st_photo=util.random_noise(photo,mode='salt')
cv.imshow("salt",noise_st_photo)
#椒噪声
noise_pr_photo=util.random_noise(photo,mode='pepper')
cv.imshow("pepper",noise_pr_photo)
#均匀噪声
noise_se_photo=util.random_noise(photo,mode='speckle')
cv.imshow("speckle",noise_se_photo)
cv.waitKey(0)
cv.destroyAllWindows()
