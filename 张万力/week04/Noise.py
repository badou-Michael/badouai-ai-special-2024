import cv2
from skimage import util
'''
调用噪声接口
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
默认是高斯

'''


img = cv2.imread("../lenna.png",0)  # step1 读取图片
poisson_img=util.random_noise(img,mode='poisson') #调用组件库的噪声方面，添加泊松噪声

cv2.imshow("source", img)
cv2.imshow("poisson",poisson_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
