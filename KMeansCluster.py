"""
Partitional clustering
In the code below, we used the following:
- Kmeans class from sklearn. cluster.
    n_clusters parameter determines the number of clusters in the data
- .fit(X) method of Kmeans
    determines clusters and their centers according to data X
- .labels_ attribute of KMeans class
    stores cluster numbers for each sample of train
    data(0 cluster, 1 cluster, 2 cluster,...)
- .cluster_centers_attribute of KMeans class
    stores cluster centers coordinates fitted by the algorithm
- .predict() method of Kmeans class
    is used to predict labels of new points
"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# Global configuration
CRED = '\033[42m'
CEND = '\033[0m'

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})

# Create toy dataset to show K-means clustering model
X = np.array([[1, 3], [2, 1], [1, 5], [8, 4], [11, 3],
              [15, 0], [6, 1], [10, 3], [3, 7], [4, 5], [12, 7]])

# Fit K-means model for 2 clusters
kmeans = KMeans(n_clusters=2).fit(X)

# Print labels for train data
print(f'{CRED}Train labels are:{CEND}', kmeans.labels_, end='\n\n')

# Print coordinates of cluster centers
print(f'{CRED}Cluster centers are:{CEND}', kmeans.cluster_centers_, end='\n\n')

# Provide predictions for new data
predicted_labels = kmeans.predict([[10, 5], [4, 2], [3, 3], [6, 3]])
print(f'{CRED}Predicted labels are:{CEND}', predicted_labels, end='\n\n')

# Visualize the results of clustering
fig, axes = plt.subplots(1, 2)
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='tab20b')
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=100)
axes[0].set_title('Train data points')

# Visualize prediction results
axes[1].scatter([10, 4, 3, 6], [5, 2, 3, 3], c=predicted_labels, s=50,
                cmap='tab20b')
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=100)
axes[1].set_title('Test data points')

plt.show()
plt.style.use('default')
