import cv2
import numpy as np
# from image_grid import ImageGrid
import matplotlib.pyplot as plt
import seaborn_image as isns
from sklearn.decomposition import PCA
# Import our K-means model
from k_means import KMeans

# PCA + k_means 视频关键帧提取

# 读取视频文件中的所有帧。
# 将帧转换为灰度图像，并按比例缩小以加快PCA（主成分分析）的速度。
# 使用PCA将帧降维到2D空间，以便于可视化。

# video_path = "../local_files/k_means/bees_original.mp4"
video_path = "simple.mp4"
# First read the video into memory
reader = cv2.VideoCapture(video_path)

frames = []
while True:
    print(f"Reading frame: {len(frames)}", end="\r")
    ret, frame = reader.read(cv2.IMREAD_GRAYSCALE)
    if ret:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayscale)
    else:
        break

frames = np.array(frames)

# For the sake of PCA speed, we're going to downscale the frames by a factor of 4.
new_shape = np.array(frames.shape)
new_shape[1:] = new_shape[1:] // 4
frames_downsized = np.zeros(new_shape, dtype=np.uint8)
# 下采样
for i in range(frames.shape[0]):
    print(f"Resizing frame: {i+1}/{frames.shape[0]}", end="\r")
    frames_downsized[i] = cv2.resize(frames[i], frames_downsized.shape[1:]).T

print("Flattening...")
# 展平操作
frames_downsized_flattened = frames_downsized.reshape(frames.shape[0], -1)

print("Decomposing...")
# PCA
pca = PCA(n_components=2)
frames_2dim = pca.fit_transform(frames_downsized_flattened)  # then fit transform the frames

# Visualize the decomposed frames. The colors progress with time
colors = plt.get_cmap("plasma")(np.linspace(0, 1, frames_2dim.shape[0]))
plt.scatter(frames_2dim[:, :1], frames_2dim[:, 1:], c=colors)
plt.show()
# Now we can cluster the video with our model to identify unique frames
#初始化KMeans对象，设置聚类数为3。
# 对PCA降维后的帧进行聚类。
clusters = 2
kmeans = KMeans(n_clusters=2)
kmeans.fit(frames_2dim)
labels = kmeans.labels
centroid_positions = kmeans.cluster_centers
print(kmeans.centroid_history)
# stable = True
# for i in range(1, len(kmeans.centroid_history)):
#     if np.abs(kmeans.centroid_history[i] - kmeans.centroid_history[i - 1]) < threshold:
#
#         break
#
# if stable:
#     print("聚类中心已经稳定，算法可能已经收敛。")
# else:
#     print("聚类中心仍然在变化，算法可能需要更多迭代。")
# while len(kmeans.centroid_history) != 9:
#
#     clusters = 3
#     kmeans = KMeans(n_clusters=3)
#     kmeans.fit(frames_2dim)
#     labels = kmeans.labels
#     centroid_positions = kmeans.cluster_centers
#     print(len(kmeans.centroid_history))

# Visualize the frame clusters
colors = plt.get_cmap("rainbow")(np.linspace(0, 1, clusters))
colors = np.flip(colors, axis=0)
data_colors = colors[labels]

plt.scatter(frames_2dim[:, :1], frames_2dim[:, 1:], c=data_colors)
plt.scatter(
    centroid_positions[:, 0],
    centroid_positions[:, 1],
    s=50,
    c=colors,
    edgecolors="black"
)
plt.show()

# Now we can select our ten frames from the clusters.
def find_closest_index(input_data, centroid):
    """Return the value in the input_data that is closet to the centroid position."""
    diff = np.abs(input_data - centroid).sum(axis=1)

    closest_index = np.argmin(diff)

    return closest_index


label_frame_indices = []
for i in range(clusters):
    input_data = frames_2dim[labels == i]
    input_indices = np.argwhere(labels == i)
    relative_index = find_closest_index(input_data, centroid_positions[i])
    abs_index = input_indices[relative_index]
    label_frame_indices.append(abs_index[0])


# Visualize the selected frames
selected_frames = frames[label_frame_indices]
# 将 selected_frames 转换为正确的形状
# selected_frames_reshaped = np.array(selected_frames).squeeze()
selected_frames_reshaped = selected_frames
img = selected_frames_reshaped[0]
img_rotated_180 = np.flipud(img)  # 先垂直翻转
img_rotated_180 = np.fliplr(img_rotated_180)
cv2.imshow("select_1", selected_frames_reshaped[0])
cv2.imshow("select_2", selected_frames_reshaped[1])
cv2.waitKey(0)

# 裁剪视频
frame_categories = labels.tolist()
print(frame_categories)
# 假设我们有一个包含0和1的列表
# 寻找第一个1出现的索引
find_x = None
if frame_categories[0] == 0:
    find_x = 1
else:
    find_x = 0
print("find_x :", find_x)
first_one_index = frame_categories.index(find_x)
#
# 寻找最后一个1出现的索引
# 反向切片列表，从后向前查找1
last_one_index = len(frame_categories) - 1 - frame_categories[::-1].index(find_x)
print(f"删除前{first_one_index}帧和后{last_one_index}帧")
