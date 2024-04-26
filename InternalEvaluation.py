"""
Internal metrics in clustering
    are used to evaluate the quality of clustering results based on the data
    without using any external information or labels. These metrics provide a
    quantitative measure of how well a clustering algorithm has grouped the
    data points into clusters based on the intrinsic characteristics of the
    data: especially intra- and inter-cluster distances.

Most commonly used internal metrics
    - Silhouette score
        measures how well a data point fits into its assigned cluster compared
        to other clusters.
        The silhouette coefficient varies between -1 to 1, with:
            * -1 indicating that the data point isn’t assigned to the right
                 cluster;
            *  0 indicating that the clusters are overlapping;
            *  1 indicates that the cluster is dense and well-separated (thus
                 the desirable value).
    - Davies-Bouldin Index (DBI)
        is an internal clustering evaluation metric that measures the quality
        of clustering by considering both the separation between clusters and
        the compactness of clusters.
        A lower DBI value indicates better clustering performance, indicating
        that the clusters are well-separated and compact.
"""
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             MeanShift, DBSCAN)
from sklearn.datasets import make_moons, make_blobs, make_circles

import matplotlib.pyplot as plt
import numpy as np
import random
# import tensorflow as tf


# Setting the seed to make the process reproducible
np.random.seed(42)
random.seed(42)
# tf.random.set_seed(42)

# Global configuration
CRED = '\033[42m'
CEND = '\033[0m'

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})

plt.figure()
plt.suptitle('Silhoutte and Davies-Bouldin Index Scores')

# Silhouette score
datasets = {
    'Circles': make_circles(n_samples=500, factor=0.2),
    'Blobs': make_blobs(n_samples=500, centers=2),
    'Moons': make_moons(n_samples=500),
}

models = {
    'KMeans': KMeans(n_clusters=2),
    'Agglomerative': AgglomerativeClustering(n_clusters=2),
    'MeanShift': MeanShift(bandwidth=2),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=20),
}

rows, cols = len(models), len(datasets)
for row, model_name in enumerate(models):
    for col, data_name in enumerate(datasets):
        # Load dataset
        X, y = datasets[data_name]

        cluster = models[model_name].fit(X)
        try:
            silhoutte = str(round(silhouette_score(X, cluster.labels_), 3))
        except ValueError:
            silhoutte = '0'
        try:
            dbi = str(round(davies_bouldin_score(X, cluster.labels_), 3))
        except ValueError:
            dbi = 'Err'
        title = f'{data_name} data cluster with {model_name}\nSilhouette is: {silhoutte}\nDBI is: {dbi}'  # noqa

        plt.subplot(rows, cols, (row+1)*cols - (cols-col) + 1)
        plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='tab20b')
        plt.title(title)

# Display the plots
plt.tight_layout()
plt.show()
