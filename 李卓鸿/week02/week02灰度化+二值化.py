import cv2
import numpy
from matplotlib import pyplot
from skimage.color import rgb2gray

pyplot.subplot(221)
img = pyplot.imread("lenna.png")
pyplot.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
pyplot.subplot(222)
img_grey = rgb2gray(img)
pyplot.imshow(img_grey,cmap='gray')
print("---img_grey----")
print(img_grey)

# 二值化
img_binary = numpy.where(img_grey >= 0.5, 1, 0)
print("-----img_binary------")
print(img_binary)
print(img_binary.shape)

pyplot.subplot(223)
pyplot.imshow(img_binary, cmap='gray')

pyplot.show()
