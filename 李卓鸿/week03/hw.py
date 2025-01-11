import numpy
import cv2
from matplotlib import pyplot

def func_nearest(img):
    height,width,channels = img.shape
    EmptyImg=numpy.zeros((1000,1000,channels),numpy.uint8)
    sh = 1000/height
    sw = 1000/width
    for i in range(1000):
        for j in range(1000):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            EmptyImg[i,j] = img[x,y]

    return EmptyImg

def func_bilinear(img,out_h,out_w):
    height, width, channels = img.shape
    EmptyImg = numpy.zeros((out_h,out_w,channels),numpy.uint8)
    sh = float( out_h / height )
    sw = float( out_w / width )
    for i in range(out_h):
        for j in range(out_w):
            for k in range(channels):
                s_height = ( i + 0.5 ) / sh - 0.5
                s_width = ( j + 0.5 ) / sw - 0.5

                s_height0 = int(numpy.floor(s_height))
                s_width0 = int(numpy.floor(s_width))
                s_height1 = min( (s_height0 + 1) , (height - 1) )
                s_width1 = min( (s_width0 + 1) , (width - 1) )

                f_x1 = (s_width1 - s_width) * img[s_height0,s_width0,k] + (s_width - s_width0) * img[s_height0,s_width1,k]
                f_x2 = (s_width1 - s_width) * img[s_height1,s_width0,k] + (s_width - s_width0) * img[s_height1,s_width1,k]

                f_xy = int( (s_height1 - s_height) * f_x1 + (s_height - s_height0) * f_x2 )

                EmptyImg[i,j,k] = f_xy

    return EmptyImg

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img2 = func_nearest(img)
    img3 = func_bilinear(img,1000,1000)

    print(img2)
    print(img2.shape)
    print('——nearest interp——')

    print(img3)
    print(img3.shape)
    print('——bilinear interp——')

    img_gray = cv2.imread('lenna.png',0)
    img_dst = cv2.equalizeHist(img_gray)

    hist = cv2.calcHist([img_dst],[0],None,[256],[0,256])
    pyplot.figure()
    pyplot.hist(img_dst.ravel(), 256)
    pyplot.show()

    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    pyplot.figure()
    pyplot.hist(img.ravel(), 256)
    pyplot.show()

    (b,g,r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    img_color = cv2.merge((bH,gH,rH))

    cv2.imshow("image",img)
    cv2.waitKey(0)

    cv2.imshow("nearest interp",img2)
    cv2.waitKey(0)

    cv2.imshow("bilinear interp",img3)
    cv2.waitKey(0)

    cv2.imshow("Histogram Equalization", numpy.hstack([img_gray, img_dst]))
    cv2.waitKey(0)

    cv2.imshow("Histogram Equaliztion rgb",img_color)
    cv2.waitKey(0)
