import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import AgglomerativeClustering


# Global configuration
CRED = '\033[42m'
CEND = '\033[0m'

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})


# this function will train agglomerative model with different linakges and
# plot the results
def check_linkage_parameter(X, y, ds_name):
    plt.figure()
    for row, linkage in enumerate(['single', 'complete', 'average']):
        agglomerative = AgglomerativeClustering(linkage=linkage,
                                                distance_threshold=0.5,
                                                n_clusters=None)
        agglomerative.fit(X)

        plt.subplot(3, 2, (row+1)*2-1)
        plt.suptitle(f'{ds_name} - Clusters with Agglomerative')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20b')
        plt.title('Real clusters')
        plt.subplot(3, 2, (row+1)*2)
        plt.scatter(X[:, 0], X[:, 1], c=agglomerative.labels_, cmap='tab20b')
        plt.title(f'Predicted using {str(linkage)} linkage')
    plt.tight_layout()


# Check clustering quality on moons dataset
X, y = make_moons(n_samples=500)
check_linkage_parameter(X, y, 'Moons')

# Check clustering quality on circles dataset
X, y = make_circles(n_samples=500)
check_linkage_parameter(X, y, 'Circles')

# Check clustering quality on blobs dataset
X, y = make_blobs(n_samples=500, cluster_std=1, centers=3)
check_linkage_parameter(X, y, 'Blobs')

plt.show()
plt.style.use('default')
