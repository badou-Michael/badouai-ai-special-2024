import cv2
import numpy as np
import matplotlib.pyplot as plt
def normalization(img):
    h,w=img.shape
    nums=[0]*256
    emptyImage = np.zeros((512,512), np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            nums[img[i][j]]+=1
    q=[0]*256
    for i in range(len(nums)):
        temp=sum(nums[:i+1])
        q[i]=int(256*temp/(h*w)-1)
    for i in range(len(img)):
        for j in range(len(img[0])):
            emptyImage[i][j]=q[img[i][j]]
    return emptyImage




def main():
    img=cv2.imread("lenna.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img=normalization(img)
    plt.figure()
    plt.hist(out_img.ravel(), 256)
    plt.show()
    cv2.imshow("Histogram Equalization", np.hstack([img, out_img]))
    cv2.waitKey(0)
if __name__=="__main__":
    main()
