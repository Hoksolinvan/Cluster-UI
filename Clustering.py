def DBSCAN(content, eps=0.5, min_samples=5):
    visited = [0] * len(content)  # 0: unvisited, 1: visited, -1: noise
    clusters = []

    def get_neighbors(index):
        neighbors = []
        for i, point in enumerate(content):
            if i != index:
                dist = ((point[0] - content[index][0]) ** 2 + (point[1] - content[index][1]) ** 2) ** 0.5
                if dist <= eps:
                    neighbors.append(i)
        return neighbors

    for i in range(len(content)):
        if visited[i] != 0:
            continue

        visited[i] = 1
        neighbors = get_neighbors(i)

        if len(neighbors) < min_samples:
            visited[i] = -1  # Mark as noise
        else:
            cluster = [i]
            queue = list(neighbors)

            while queue:
                j = queue.pop()
                if visited[j] == 0:
                    visited[j] = 1
                    j_neighbors = get_neighbors(j)
                    if len(j_neighbors) >= min_samples:
                        queue.extend(j_neighbors)
                if all(j not in c for c in clusters):  # Avoid adding duplicates
                    cluster.append(j)

            clusters.append(cluster)

    return clusters


def KMeans(content, k=3, max_iterations=100):
    import random

    centroids = random.sample(content, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in content:
            distances = [((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2) ** 0.5 for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid division by zero
                new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(content))  # Reinitialize centroid if cluster is empty

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return clusters

def Sci_kitDBSCAN(content, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    import numpy as np

    content_np = np.array(content)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(content_np)

    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise points
            clusters.append(np.where(labels == label)[0].tolist())

    return clusters

def Sci_kitKMeans(content, k=3, max_iterations=100):
    from sklearn.cluster import KMeans
    import numpy as np

    content_np = np.array(content)
    kmeans = KMeans(n_clusters=k, max_iter=max_iterations)
    kmeans.fit(content_np)

    clusters = [[] for _ in range(k)]
    for index, label in enumerate(kmeans.labels_):
        clusters[label].append(content[index])

    return clusters

def CUML_DBSCAN(content, eps=0.5, min_samples=5):
    from cuml.cluster import DBSCAN as CumlDBSCAN
    import numpy as np

    content_np = np.array(content)
    dbscan = CumlDBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(content_np)

    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise points
            clusters.append(np.where(labels == label)[0].tolist())

    return clusters

def CUML_KMeans(content, k=3, max_iterations=100):
    from cuml.cluster import KMeans as CumlKMeans
    import numpy as np

    content_np = np.array(content)
    kmeans = CumlKMeans(n_clusters=k, max_iter=max_iterations)
    kmeans.fit(content_np)

    clusters = [[] for _ in range(k)]
    for index, label in enumerate(kmeans.labels_):
        clusters[label].append(content[index])

    return clusters
    
