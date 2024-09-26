import cv2
from matplotlib import pyplot as plt
if __name__ == '__main__':
    img = cv2.imread('girls.jpg')
    (b, g, r) = cv2.split(img)
    B_H = cv2.equalizeHist(b)
    G_H = cv2.equalizeHist(g)
    R_H = cv2.equalizeHist(r)

    result = cv2.merge((B_H, G_H, R_H))

    cv2.imshow("result", result)


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixles')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    cv2.waitKey(0)
