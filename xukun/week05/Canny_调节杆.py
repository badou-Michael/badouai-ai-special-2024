import cv2


def Canny_regulate(lowThreshold):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, lowThreshold, lowThreshold * 3, apertureSize=3, L2gradient=True)
    dst = cv2.bitwise_and(img, img, mask=canny_img)
    cv2.imshow('canny result', dst)

img = cv2.imread('lenna.png')
lowThreshold = 0
max_lowThreshold = 250
cv2.namedWindow('canny result')

cv2.createTrackbar('Min threshold', 'canny result', lowThreshold, max_lowThreshold, Canny_regulate)

Canny_regulate(0)  # initialization
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()
