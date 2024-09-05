from PIL import Image
import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# 使用PIL库转换灰度图/二值图
def processPIL():
    #打开图像
    img = Image.open("lenna.png")   #返回PIL.Image.Image对象,0~255
    img = img.convert("RGB")    #确保图象是rgb模式
    #获取长宽h,w
    width, height = img.size
    
    #创建新图像
    gray_img = Image.new("L", (width, height))
    
    #遍历每个像素给新图像赋值
    for w in range(width):
        for h in range(height):
            r,g,b = img.getpixel((w,h))
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_img.putpixel((w,h), gray)
    
    #展示灰度图
    gray_img.show()
    
    #创建二值图
    binary_img = Image.new("1", (width, height))
    #设置像素点
    for w in range(width):
        for h in range(height):
            gray = gray_img.getpixel((w,h))
            if gray > 128:
                binary = 1
            else:
                binary = 0
            binary_img.putpixel((w,h), binary)
    binary_img.show()


#使用opencv实现转换
def processCV(auto = True):
    img = cv2.imread("lenna.png")   #返回numpy.ndarray对象,0~255
    height, width = img.shape[:2]   #shape是（512，512，3）的矩阵
    if auto == True:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    else:
        gray_img = np.zeros((height,width), dtype=np.uint8)  #新建numpy数组
        for h in range(height):
            for w in range(width):
                pix = img[h,w]  #pix是BGR坐标
                gray_img[h,w] = int(pix[0]*0.114 + pix[1]*0.587 + pix[2]*0.299) #注意opencv保存图像的顺序
    
        binary_img = np.zeros((height, width), dtype=np.uint8)
        for h in range(height):
            for w in range(width):
                pix = gray_img[h,w]
                if pix > 128:
                    binary_img[h,w] = 255
                else:
                    binary_img[h,w] = 0
    
    print(gray_img)
    
    cv2.imshow("image show gray", gray_img)
    cv2.imshow("binary show ", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# plt写法
def processPlt():
    plt.subplot(221)    #返回numpy.ndarray对象
    img = plt.imread("lenna.png")   #如果是PNGt图像，img是0~1之间的浮点数
    plt.imshow(img)
    print("---image lenna---")
    print(img)
    print(img.shape)
    
    gray_img = rgb2gray(img)
    print("---image gray---")
    print(gray_img)
    plt.subplot(222)
    # camp="gray"指定使用灰度彩色映射来显示图像
    plt.imshow(gray_img, cmap="gray")
    
    binary_img = np.where(gray_img>=0.5, 1, 0)
    print("---image binary---")
    print(binary_img)
    plt.subplot(223)
    plt.imshow(binary_img, cmap = "gray")
    
    plt.show()  #显示plot
    
if __name__ == "__main__":
    # processPIL()      #PIL方法
    # processCV(True)   #CV方法
    processPlt()        #PLT方法
    
    