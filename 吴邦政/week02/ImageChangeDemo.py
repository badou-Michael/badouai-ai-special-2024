from PIL import Image

#打开图片
image = Image.open("image.jpg")
#获取图片大小
width,height = image.size
#转化为灰度图
for i in range(height):
    for j in range(width):
        r,g,b = image.getpixel((j,i))
        gray_color =int(0.3 * r + 0.6 * g + 0.1 * b)
        image.putpixel((j,i), (gray_color,gray_color,gray_color))

#展示
image.show()

#二值图阈值
threshold = 128
#灰度图转化二值图
for y in range(height):
    for x in range(width):
        gray = image.getpixel((x,y))[0]
        if gray >= threshold:
            image.putpixel((x,y),(255,255,255))
        else:
            image.putpixel((x, y), (0, 0, 0))
#展示
image.show()