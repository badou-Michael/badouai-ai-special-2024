import cv2

img=cv2.imread("lenna.png")

(b,g,r)=cv2.split(img)

sb=cv2.equalizeHist(b)
sg=cv2.equalizeHist(g)
sr=cv2.equalizeHist(r)

dst = cv2.merge((sb,sg,sr))

cv2.imshow("result",dst)

cv2.waitKey(0)
