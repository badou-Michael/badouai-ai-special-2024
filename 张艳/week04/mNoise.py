import cv2
import numpy as np
from PIL import Image
from skimage import util

# 生成噪声的相关接口之 util.noise.random_noise（image, mode='gaussian', seed=None, clip=True, **kwargs）

img = cv2.imread("lenna.png")
cv2.imshow("img", img)

# gauss高斯噪声：不能设置amount
img_gauss = util.noise.random_noise(img, mode='gaussian', mean=0, var=0.01)
cv2.imshow("img_gauss", img_gauss)

# salt：盐噪声，随机将像素值变成1
# img_salt = util.noise.random_noise(img, mode='salt')  # amount=0.05
# cv2.imshow("img_salt 0.05", img_salt)
# img_salt2 = util.noise.random_noise(img, mode='salt', amount=0.1)
# cv2.imshow("img_salt 0.1", img_salt2)

# pepper：椒噪声，随机将像素值变成0或 -1，取决于矩阵的值是否带符号
# img_pepper = util.noise.random_noise(img, mode='pepper')  # amount=0.05
# cv2.imshow("img_pepper 0.05", img_pepper)
# img_pepper2 = util.noise.random_noise(img, mode='pepper', amount=0.1)
# cv2.imshow("img_pepper 0.1", img_pepper2)

# s & p：椒盐噪声
# img_sp = util.noise.random_noise(img, mode='s&p')  # amount=0.05, salt_vs_pepper=0.5
# cv2.imshow("img_sp 0.05", img_sp)
img_sp2 = util.noise.random_noise(img, mode='s&p', amount=0.1, salt_vs_pepper=0.9)
cv2.imshow("img_sp 2", img_sp2)
# img_sp3 = util.noise.random_noise(img, mode='s&p', amount=0.1, salt_vs_pepper=0.1)
# cv2.imshow("imgs&p 3", img_sp3)

# speckle：均匀噪声
# img_speckle = util.random_noise(img, mode='speckle')  # mean=0, var=0.01
# cv2.imshow("img_speckle", img_speckle)
# img_speckle2 = util.random_noise(img, mode='speckle', mean=0, var=0.2)
# cv2.imshow("img_speckle 2", img_speckle2)

# localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差噪声。
# img_localvar = util.random_noise(img, mode='localvar')
# cv2.imshow("img_localvar", img_localvar)

# poisson：泊松噪声，不能设置mean和var
# img_poisson = util.random_noise(img, mode='poisson')
# cv2.imshow("img_poisson", img_poisson)

cv2.waitKey(0)
cv2.destroyAllWindows()
