import numpy as np
import cv2

def guass_filter(image, kernel_size=5, sigma=1.5):
   height, width = image.shape[:2]
   center = kernel_size // 2
   guass_kernel = np.zeros((kernel_size, kernel_size))
   for i in range(kernel_size):
      for j in range(kernel_size):
         x = j - center
         y = j - center
         guass_kernel[i, j] =  (1 / (2 * np.pi * sigma **2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
   guass_kernel = guass_kernel / np.sum(guass_kernel)

   filter_image = np.zeros((height, width), dtype=np.uint8)
   for i in range(center, height - center):
      for j in range(center, width - center):
         patch = image[i - center:i - center + 1, j - center:j + center + 1]
         filter_image[i, j] = np.sum(patch * guass_kernel)
   return filter_image
def sobel_detect(gray_img : np.ndarray):
   sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
   sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
   gradient_magnitude = np.zeros_like(gray_img, dtype=np.float64)
   gradient_direction = np.zeros_like(gray_img, dtype=np.float64)
   height, width = gray_img.shape
   for i in range(1, height - 1):
      for j in range(1, width - 1):
         region = gray_img[i - 1: i + 2, j - 1: j + 2]
         Gx = np.sum(region * sobel_x)
         Gy = np.sum(region * sobel_y)
         gradient_magnitude[i,j] = np.sqrt(Gx**2 + Gy**2)
         gradient_direction[i,j] = np.arctan2(Gy, Gx)
   gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)
   return gradient_magnitude, gradient_direction 



def nms(gradient_magnitude, gradient_direction):
   height, width = gradient_magnitude.shape
   suppressed = np.zeros((height, width), dtype=np.uint8)
   for i in  range(1, height - 1):
      for j in range(1, width - 1):
         angle = gradient_direction[i, j]
         if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
            n1, n2 = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
         elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
            n1, n2 = gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]
         elif (67.5 <= angle < 112.5) or (-122.5 <= angle < -67.5):
            n1, n2 = gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]
         else:
            n1, n2 = gradient_magnitude[i - 1, j-1], gradient_magnitude[i +1 , j+1]
         if gradient_magnitude[i, j] >= n1 and gradient_magnitude[i, j] >= n2:
            suppressed[i, j] = gradient_magnitude[i, j]
   return suppressed

def double_threshold(suppressed_magnitude, low, high):
   height, width = suppressed_magnitude.shape
   strong_edge = np.zeros((height, width), dtype=np.uint8)
   weak_edge = np.zeros((height, width), dtype=np.uint8)
   for i in range(height):
      for j in range(width):
         if suppressed_magnitude[i, j] >= high:
            strong_edge[i, j] = 255
         elif suppressed_magnitude[i, j] >= low:
            weak_edge[i, j] = 255
   return strong_edge, weak_edge

def edge_tracking_by_hysteresis(strong_edge, weak_edge):
   height, width = strong_edge.shape
   final_edge = strong_edge.copy()
   for i in range(1, height - 1):
      for j in range(1, width - 1):
         if weak_edge[i, j]:
            if np.any(strong_edge[i-1:i+2, j-1:j+2]):
               final_edge[i, j] = 255
            else:
               final_edge[i, j] = 0
   return final_edge


if __name__ == '__main__':
   img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
   guass_data = guass_filter(img)
   cv2.imshow('guass', guass_data)
   magnti, direct = sobel_detect(guass_data)
   cv2.imshow('sobel', magnti)
   nms_data = nms(magnti, direct)
   cv2.imshow('nms_data', nms_data)
   strong, weak = double_threshold(nms_data, 10, 50)
   final_edge = edge_tracking_by_hysteresis(strong, weak)
   print(final_edge)
   cv2.imshow('final_edge', final_edge)

   edges = cv2.Canny(img, threshold1=10, threshold2=50)
   cv2.imshow('canny', edges)
   cv2.waitKey(0)