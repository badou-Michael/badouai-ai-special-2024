import cv2
from skimage import util
import numpy as np

img = cv2.imread("lenna.png",0)
# 注意这里先把图像从0-255归一化成0-1再添加高斯噪声的，
# 所以方差的参数的数量级也要相应缩小
gaussion_img = util.random_noise(img, mode='gaussian', seed=None, clip=True, mean=0.0078, var=0.0078)
poisson_img = util.random_noise(img, mode="poisson", seed=None, clip=True)
# amount代表椒盐噪声所占比例
salt_img = util.random_noise(img, mode='salt', amount=0.01)
pepper_img = util.random_noise(img, mode='pepper', amount=0.01)
# salt_vs_pepper
sp_img = util.random_noise(img, mode='s&p', amount=0.02, salt_vs_pepper=0.1)
# 高斯分布的加性噪声
# 可以指定每个点的噪声的方差
local_variance = np.random.uniform(0.001,0.01, size = img.shape) #生成均匀分布的噪声
localvar_img = util.random_noise(img, mode='localvar', seed=False, clip=True, local_vars=local_variance)

cv2.imshow("ori", img)
cv2.imshow("gaussion", gaussion_img)
cv2.imshow("poisson", poisson_img)
cv2.imshow("salt", salt_img)
cv2.imshow("pepper", pepper_img)
cv2.imshow("s&p", sp_img)
cv2.imshow("localvar", localvar_img)
cv2.waitKey()
cv2.destroyAllWindows()