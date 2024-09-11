from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# 读取原始图片
image = imread('lenna.png')

# 灰度化
gray_image = rgb2gray(image)
imsave('lenna_gray.png', (gray_image * 255).astype('uint8'))

# 二值化
threshold = threshold_otsu(gray_image)
binary_image = gray_image > threshold
imsave('lenna_binary.png', (binary_image.astype('uint8') * 255))
