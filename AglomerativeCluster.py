"""
Agglomerative clustering is a hierarchical clustering algorithm used
in machine learning and data mining: it groups similar data points into
nested clusters based on their pairwise distances.
AgglomerativeClustering class has no implementation for .predict() method:
we have to train the model every time we want to cluster new data.
"""
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Firstly, we will create our dataset
X, y = make_blobs(n_samples=500, cluster_std=1, centers=4, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.matmul(X, transformation)  # Matrix product of two arrays.

# In this line we will specify parameters of our Agglomerative model
agglomerative = AgglomerativeClustering(linkage='single',
                                        distance_threshold=0.6,
                                        n_clusters=None)
# Training our model
agglomerative.fit(X_aniso)

# Providing visualization of results
plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=agglomerative.labels_, s=50, cmap='tab20b')
plt.title('Clustered data')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='tab20b')
plt.title('Real data')
plt.show()
