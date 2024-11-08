import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Basketball player data (assists per minute, points per minute)
X = np.array([
    [0.0888, 0.5885], [0.1399, 0.8291], [0.0747, 0.4974], [0.0983, 0.5772],
    [0.1276, 0.5703], [0.1671, 0.5835], [0.1306, 0.5276], [0.1061, 0.5523],
    [0.2446, 0.4007], [0.1670, 0.4770], [0.2485, 0.4313], [0.1227, 0.4909],
    [0.1240, 0.5668], [0.1461, 0.5113], [0.2315, 0.3788], [0.0494, 0.5590],
    [0.1107, 0.4799], [0.1121, 0.5735], [0.1007, 0.6318], [0.2567, 0.4326],
    [0.1956, 0.4280]
])

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # Set random_state for reproducibility
y_pred = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))  # Set figure size for better visualization
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='x', s=50, cmap='viridis') # Use a colormap

# Add labels and title
plt.xlabel("Assists per Minute")
plt.ylabel("Points per Minute")
plt.title("KMeans Clustering of Basketball Player Data")

# Add a legend
handles, labels = scatter.legend_elements(prop="colors")
plt.legend(handles, ["Cluster 0", "Cluster 1", "Cluster 2"], title="Clusters") # Dynamic legend


plt.grid(True, alpha=0.3) # Add a subtle grid
plt.show()