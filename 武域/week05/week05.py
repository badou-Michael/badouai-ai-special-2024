import numpy as np
import matplotlib.pyplot as plt

# Gaussian filter
def gaussian_kernel(size=5, sigma=1.52):  # kernel size:5, sigma val: 1.52
    kernel = np.zeros((size, size))  # create empty kernel
    center = size // 2  # center of kernel (floor division)
    sum_val = 0

    for i in range(size):  # calculate kernel
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]

    return kernel / sum_val  # normalize kernel

def apply_filter(image, kernel):
    smoothed_image = np.zeros_like(image, dtype=np.float64)
    padded_image = np.pad(image, ((kernel.shape[0] // 2, kernel.shape[1] // 2)), mode='constant')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            smoothed_image[i, j] = np.sum(kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])

    return smoothed_image

# Sobel filters for gradient detection
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

SOBEL_Y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])

# Gradient magnitude and direction detection
def gradient_mag_dir(image):
    Gx = apply_filter(image, SOBEL_X)
    Gy = apply_filter(image, SOBEL_Y)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    dir = np.arctan2(Gy, Gx) * 180 / np.pi
    dir = (dir + 180) % 180  # normalize
    return mag, dir

# Non-maximum suppression
def non_max_sup(mag, dir):
    nms = np.zeros_like(mag)

    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):
            q, r = 255, 255
            angle = dir[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                nms[i, j] = mag[i, j]
            else:
                nms[i, j] = 0
    return nms

# Thresholding
def threshold(image, low, high):
    res = np.zeros_like(image)
    strong = 255
    weak = 50
    
    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res

# Edge tracking by hysteresis
def edge_tracking_by_hysteresis(image, weak=50, strong=255):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == weak:
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                    or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                    or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

# Canny edge detection
def canny_edge_detection(image, low_threshold, high_threshold, kernel_size=5, sigma=1.52):
    # Ensure the image is scaled between 0 and 255
    if np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Apply Gaussian blur
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = apply_filter(image, kernel)

    # Compute gradient magnitude and direction
    mag, dir = gradient_mag_dir(blurred_img)

    # Apply non-maximum suppression
    nms = non_max_sup(mag, dir)

    # Apply double threshold
    thresh = threshold(nms, low_threshold, high_threshold)

    # Perform edge tracking by hysteresis
    edge = edge_tracking_by_hysteresis(thresh)
    
    return edge

# Load the image
image = plt.imread('gray_lenna.png')

# Convert to grayscale if it's not already
if len(image.shape) == 3:
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Scale the grayscale image if needed
if np.max(image) <= 1.0:
    image = (image * 255).astype(np.uint8)

# Debug: Display intermediate steps
plt.figure(figsize=(12, 6))

# Gaussian blurred image
blurred_img = apply_filter(image, gaussian_kernel(5, 1.52))
plt.subplot(1, 3, 1)
plt.imshow(blurred_img, cmap='gray')
plt.title('Blurred Image')

# Gradient magnitude
mag, _ = gradient_mag_dir(blurred_img)
plt.subplot(1, 3, 2)
plt.imshow(mag, cmap='gray')
plt.title('Gradient Magnitude')

# Canny edge detection
edges = canny_edge_detection(image, low_threshold=50, high_threshold=150, kernel_size=5, sigma=1.52)
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.show()
