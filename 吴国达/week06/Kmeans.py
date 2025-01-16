import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("../images/lenna.png", 0)
    data = img.reshape((img.shape[0] * img.shape[1], 1))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    retval, bestLabels, centers = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dst = bestLabels.reshape((img.shape[0], img.shape[1]))
    print(dst)
    cv2.imshow("dst", dst/4)
    cv2.waitKey(0)
