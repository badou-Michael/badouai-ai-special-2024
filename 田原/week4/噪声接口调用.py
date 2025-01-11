import cv2
from skimage import util
import numpy as np

img = cv2.imread("lenna.png",0)
#高斯噪声
gaussion_img = util.random_noise(img, mode='gaussian', seed=None, clip=True, mean=0.0078, var=0.0078)
#泊松噪声
poisson_img = util.random_noise(img, mode="poisson", seed=None, clip=True)
# amount代表椒盐噪声所占比例
salt_img = util.random_noise(img, mode='salt', amount=0.01)
pepper_img = util.random_noise(img, mode='pepper', amount=0.01)
# 椒盐噪声
sp_img = util.random_noise(img, mode='s&p', amount=0.02, salt_vs_pepper=0.1)

cv2.imshow("origin", img)
cv2.imshow("gaussion", gaussion_img)
cv2.imshow("poisson", poisson_img)
cv2.imshow("salt", salt_img)
cv2.imshow("pepper", pepper_img)
cv2.imshow("s&p", sp_img)
cv2.waitKey()
cv2.destroyAllWindows()
