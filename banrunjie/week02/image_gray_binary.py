import cv2
import numpy as np


if __name__ == "__main__":
    im_gray = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
    cv2.imshow("image show gray",im_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im_binary=cv2.convertScaleAbs(np.where(im_gray > 127, 255, 0))
    cv2.imshow("image show binary",im_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

