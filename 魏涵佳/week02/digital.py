from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('../imgs/wallpaper_green.jpg')
if img is not None:
    print(img.shape)  # (height, width, channels)
else:
    print("Image not loaded")

# manually convert to grayscale
h, w = img.shape[:2]

img_gray = np.zeros((h, w), dtype=img.dtype)
for i in range(h):
    for j in range(w):
        # constrain the pixel value to be between 0 and 255
        img_gray[i,j] = np.clip(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3, 0, 255)

# ----------------------------- use cv.imshow to display the image -----------------------------
cv2.namedWindow('gray_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('gray_img', 600, 600)
cv2.imshow('gray_img', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------- use matplotlib to display the image -----------------------------
plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')  # disable axis

# if use plt to show the cv2 loaded image, the color should be converted to RGB first.
plt.subplot(222)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# ----------------------------- use skimage to convert to grayscale -----------------------------
img_rgb = plt.imread('../imgs/wallpaper_green.jpg')
img_gray_sk = rgb2gray(img_rgb)
plt.subplot(223)
plt.imshow(img_gray_sk, cmap='gray')
plt.title('Grayscale Image using skimage')
plt.axis('off')

# ----------------------------- use np to convert to binaryscale -----------------------------
img_binary = np.where(img_gray > 128, 255, 0)
plt.subplot(224)
plt.imshow(img_binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.show()
