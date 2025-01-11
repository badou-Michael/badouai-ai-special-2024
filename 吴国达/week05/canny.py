import cv2

if __name__ == "__main__":
    img = cv2.imread("../images/lenna.png")
    canny = cv2.Canny(img, 100, 300)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
