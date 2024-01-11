import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, X_test, k):
    y_pred = []

    for test_point in X_test:
        # Calculate distances between the test point and all training points
        distances = [euclidean_distance(test_point, x_train) for x_train in X_train]

        # Sort distances and return the indices of k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Determine the labels of these nearest neighbors
        k_nearest_labels = [y_train[i] for i in k_indices]

        # Majority vote for the predicted label
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])

    return np.array(y_pred)
