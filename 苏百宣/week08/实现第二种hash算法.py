# 实现第二种hash算法
# author: 苏百宣
import cv2


def dhash(img):
    img = cv2.resize(img, (9, 10), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str=''
    for i in range(9):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

# 加载图片
img = cv2.imread('/Users/ferry/Desktop/八斗作业/week08/sww1028.jpg')

# 检查图片是否加载成功
if img is None:
    raise ValueError("Image not found or unable to load!")

# 计算并打印哈希值
hash_value = dhash(img)
print(f"Hash Value: {hash_value}")

# 显示图片
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
