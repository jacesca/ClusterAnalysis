import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans


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
        'Xy': make_blobs(n_samples=500, cluster_std=1, centers=3),
        'n_clusters': 3,
    },
    'Moons': {
        'Xy': make_moons(n_samples=500),
        'n_clusters': 2,
    },
    'Circles': {
        'Xy': make_circles(n_samples=500),
        'n_clusters': 2,
    }
}

plt.figure()
plt.suptitle('KMeans Clusters')
for row, data_name in enumerate(datasets.keys()):
    X, y = datasets[data_name]['Xy']
    kmeans = KMeans(n_clusters=datasets[data_name]['n_clusters']).fit(X)

    plt.subplot(3, 2, (row+1)*2-1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20b')
    plt.title(f'{data_name} Real clusters')
    plt.subplot(3, 2, (row+1)*2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab20b')
    plt.title('Predicted clusters')

plt.tight_layout()
plt.show()
plt.style.use('default')
