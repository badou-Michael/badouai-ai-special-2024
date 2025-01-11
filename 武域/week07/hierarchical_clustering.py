import numpy as np
def single_linkage_distance(cluster1, cluster2):
    min_distance = np.inf
    for point1 in cluster1:
        for point2 in cluster2:
            dist = np.linalg.norm(np.array(point1) - np.array(point2))
            if dist < min_distance:
                min_distance = dist
    print(f"Distance between {cluster1} and {cluster2} is {min_distance}")
    return min_distance

def ward_linkage_distance(cluster1, cluster2):
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    combined_cluster = np.vstack((cluster1, cluster2))
    combined_centroid = np.mean(combined_cluster, axis=0)
    ssq1 = np.sum((cluster1 - centroid1) ** 2)
    ssq2 = np.sum((cluster2 - centroid2) ** 2)
    ssq_combined = np.sum((combined_cluster - combined_centroid) ** 2)
    variance_increase = ssq_combined - (ssq1 + ssq2)
    print(f"Distance between {cluster1} and {cluster2} is {variance_increase}")
    return variance_increase

def hierarchical_clustering(data, numclusters):
    clusters = [[point] for point in data]
    print(f"Initial clusters: {clusters}")
    while len(clusters) > numclusters:
        min_distance = np.inf
        cluster_to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = ward_linkage_distance(clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster_to_merge = (i, j)

        cluster1, cluster2 = cluster_to_merge
        print(f"Merging clusters {clusters[cluster1]} and {clusters[cluster2]} with distance {min_distance}")
        clusters[cluster1].extend(clusters[cluster2])
        clusters.pop(cluster2)
        print(f"Clusters after merging: {clusters}")
    return clusters

if __name__ == "__main__":
    data = [[1,2],[3,2],[4,4],[1,2],[1,3]]
    numclusters = 2
    clusters = hierarchical_clustering(data, numclusters)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
