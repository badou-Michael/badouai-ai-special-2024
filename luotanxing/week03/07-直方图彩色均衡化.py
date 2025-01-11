import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../week02/lenna.png')
#通道为gbr
b,g,r = cv2.split(img)

bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

result = cv2.merge((bh, gh, rh))
cv2.imshow('result',result)
cv2.waitKey(0)