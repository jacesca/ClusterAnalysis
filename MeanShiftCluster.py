"""
Mean shift is the most simple density-based clustering algorithm.
In MeanShift class you can use .predict() method to make predictions
based on an already trained model.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import MeanShift


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
        'Xy': make_blobs(n_samples=500, cluster_std=1, centers=4),
        'bandwidth': 2,
    },
    'Moons': {
        'Xy': make_moons(n_samples=500),
        'bandwidth': 0.7,
    },
    'Circles': {
        'Xy': make_circles(n_samples=500),
        'bandwidth': 0.1,
    }
}

plt.figure()
plt.suptitle('Mean Shift Clusters')
for row, data_name in enumerate(datasets.keys()):
    X, y = datasets[data_name]['Xy']
    moons_clustering = MeanShift(bandwidth=datasets[data_name]['bandwidth']).fit(X)  # noqa

    plt.subplot(3, 2, (row+1)*2-1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20b')
    plt.title(f'{data_name} Real clusters')
    plt.subplot(3, 2, (row+1)*2)
    plt.scatter(X[:, 0], X[:, 1], c=moons_clustering.labels_, cmap='tab20b')
    plt.title('Predicted clusters')

plt.tight_layout()
plt.show()
plt.style.use('default')
