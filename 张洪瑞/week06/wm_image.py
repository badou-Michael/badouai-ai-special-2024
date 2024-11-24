import cv2
import numpy as np


pth = "photo1.jpg"
img = cv2.imread(pth)
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
warp_Matrix = cv2.getPerspectiveTransform(src, dst)
warp_img = cv2.warpPerspective(img, warp_Matrix, (500, 500))
cv2.imshow("Img", img)
cv2.imshow("Warp_Img", warp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
