import cv2
#实现灰度化
image=cv2.imread("lena.png")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#实现二值化
ret,binary_image=cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Binary Image',binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
