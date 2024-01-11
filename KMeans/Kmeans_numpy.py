import numpy as np

def initialize_centroids_plusplus(points, k):
    """ Initialize centroids using K-means++ method """
    centroids = [points[np.random.randint(points.shape[0])]]
    for _ in range(1, k):
        distances = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in points])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()

        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                i_selected = i
                break

        centroids.append(points[i_selected])

    return np.array(centroids)

def closest_centroid(points, centroids):
    """ Return an array containing the index to the nearest centroid for each point """
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """ Return the new centroids assigned from the points closest to them """
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def kmeans_plusplus(points, k, max_iterations=100):
    """ K-means++ implementation """
    centroids = initialize_centroids_plusplus(points, k)
    for _ in range(max_iterations):
        closest = closest_centroid(points, centroids)
        new_centroids = move_centroids(points, closest, centroids)

        # Check for convergence (if centroids don't change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, closest