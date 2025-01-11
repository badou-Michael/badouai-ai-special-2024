import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/zhouzhaoyu/Desktop/ai/photo1.jpg")
# img = plt.imread("/Users/zhouzhaoyu/Desktop/ai/photo1.jpg")
# plt.imshow(img)
# plt.show()
trans_img = img.copy()
src = np.float32([[208,156],
                  [518,288],
                  [19,610],
                  [345,738]])
tgt = np.float32([[0,400],
                  [200,400],
                  [0,0],
                  [200,0]])
transMat = cv2.getPerspectiveTransform(src, tgt)
trans_img = cv2.warpPerspective(img, transMat, (200,400))
cv2.imshow("original image", img)
cv2.imshow("transformed image", trans_img)
cv2.waitKey(0)