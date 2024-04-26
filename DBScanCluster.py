"""
Mean shift is the most simple density-based clustering algorithm.
DBSCAN class has no implementation for .predict() method, so we can
not provide clustering for new points without retraining the model.
DBSCAN algorithm allocates noise to a separate cluster with -1 label.
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import DBSCAN


# Global configuration
CRED = '\033[42m'
CEND = '\033[0m'

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})


# Let's use check_clustering_quality function on blobs,
# moons and circles datasets
datasets = {
    'Blobs': {
        'Xy': make_blobs(n_samples=500, centers=3),
        'params': {'eps': 0.85, 'min_samples': 20},
    },
    'Moons': {
        'Xy': make_moons(n_samples=500, noise=0.05, random_state=30),
        'params': {'eps': 0.3},
    },
    'Circles': {
        'Xy': make_circles(n_samples=2000, noise=0.1, factor=0.2),
        'params': {'eps': 0.1, 'min_samples': 5},
    }
}

plt.figure()
plt.suptitle('DBSCAN Clusters')
for row, data_name in enumerate(datasets.keys()):
    X, y = datasets[data_name]['Xy']
    moons_clustering = DBSCAN(**datasets[data_name]['params']).fit(X)
    detected_clusters = moons_clustering.labels_
    # cleaned_X = np.delete(X, np.where(detected_clusters == -1), axis=0).reshape(-1, 2)  # noqa
    # outliers_X = np.delete(X, np.where(detected_clusters != -1), axis=0).reshape(-1, 2)  # noqa
    cleaned_X = np.delete(X, np.where(detected_clusters == -1), axis=0)  # noqa
    outliers_X = np.delete(X, np.where(detected_clusters != -1), axis=0)  # noqa
    plt.subplot(3, 4, (row+1)*4-3)
    plt.title(f'{data_name} Real clusters')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20b')
    plt.subplot(3, 4, (row+1)*4-2)
    plt.title('Predicted clusters')
    plt.scatter(X[:, 0], X[:, 1], c=detected_clusters, cmap='tab20b')
    plt.subplot(3, 4, (row+1)*4-1)
    plt.title('Outliers Detected')
    plt.scatter(outliers_X[:, 0], outliers_X[:, 1], c='darkblue')
    plt.subplot(3, 4, (row+1)*4)
    plt.title('Cleaned clusters')
    plt.scatter(cleaned_X[:, 0], cleaned_X[:, 1], c=detected_clusters[detected_clusters!=-1], cmap='tab20b')  # noqa

plt.tight_layout()
plt.show()
plt.style.use('default')
