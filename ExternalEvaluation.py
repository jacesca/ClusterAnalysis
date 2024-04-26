"""
External evaluation for clustering algorithms
    is a method of evaluating the performance of a clustering algorithm by
    comparing its results to a known set of class labels or ground truth.

Most commonly used external metrics
    - The Rand Index (RI)
        measures the similarity between two clusterings or partitions and
        is often used as an external evaluation metric in clustering.
        The Rand Index can vary between 0 and 1, where 0 indicates that the
        two clusterings are completely different, and 1 indicates that the two
        clusterings are identical.
    - Mutual Information (MI)
        measures the amount of information shared by the predicted and true
        clusterings based on the concept of entropy.
        The Mutual Information varies between 0 and 1, where 0 indicates that
        the predicted clustering is completely different from the true
        clustering, and 1 indicates that the predicted clustering is identical
        to the true clustering.
    - Homogeneity
        measures the degree to which each cluster contains only data points
        that belong to a single class or category based on conditional entropy.
        The homogeneity score ranges from 0 to 1, with 1 indicating perfect
        homogeneity.
        Homogeneity is the best of all the considered metrics: it determines
        both good and bad clustering equally well.
"""
from sklearn.metrics import rand_score, mutual_info_score, homogeneity_score
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
            predicted_classes = cluster.predict(X)
        except AttributeError:
            predicted_classes = cluster.labels_
        assert (cluster.labels_ == predicted_classes).all()

        randscore = str(round(rand_score(y, predicted_classes), 3))
        mutualscore = str(round(mutual_info_score(y, predicted_classes), 3))
        homogeneityscore = str(round(homogeneity_score(y, predicted_classes), 3))  # noqa
        title = f'{data_name} data cluster with {model_name}\nRand Score is: {randscore}\nMutual Information (MI) is: {mutualscore}\nHomogeneity is: {homogeneityscore}'  # noqa

        plt.subplot(rows, cols, (row+1)*cols - (cols-col) + 1)
        plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='tab20b')
        plt.title(title)

# Display the plots
plt.tight_layout()
plt.show()
