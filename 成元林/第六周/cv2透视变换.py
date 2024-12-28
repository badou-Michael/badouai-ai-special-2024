import cv2
import numpy as np

def transImg(srcimg,src,dst):
    t_img = np.copy(srcimg)
    wrapMatrix = cv2.getPerspectiveTransform(src,dst)
    result = cv2.warpPerspective(t_img,wrapMatrix,(400,550))
    return result

if __name__ == '__main__':
    src = np.float32([[208,153],[518,286],[16,604],[341,733]])
    dst = np.float32([[0,0],[400,0],[0,550],[400,550]])
    srcimg = cv2.imread("photo1.jpg")
    result = transImg(srcimg,src,dst)
    cv2.imshow("srcimg",srcimg)
    cv2.imshow("result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()