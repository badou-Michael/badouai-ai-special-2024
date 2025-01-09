import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../images/photo1.jpg")
    print(img.shape)
    copy = img.copy()
    src = np.float32([[207, 159], [517, 293], [17, 602], [343, 731]])
    dst = np.float32([[0, 0], [350, 0], [0, 500], [350, 500]])
    transform = cv2.getPerspectiveTransform(src, dst)
    perspective = cv2.warpPerspective(copy, transform, (350, 500))
    cv2.imshow("img", img)
    cv2.imshow("perspective", perspective)
    cv2.waitKey(0)
