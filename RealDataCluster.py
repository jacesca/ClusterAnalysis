from sklearn.datasets import load_iris
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             MeanShift, DBSCAN)
import matplotlib.pyplot as plt


# Global configuration
CRED = '\033[42m'
CEND = '\033[0m'

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})

# REading the data
classes = list(load_iris().target_names)
X_iris, y_iris = load_iris(return_X_y=True)
X_iris = X_iris[:, [0, 2]]  # Selecting Length of sepals and Length of petals

# Let's apply different clusters models
datasets = {
    'KMeans': KMeans(n_clusters=3),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'MeanShift': MeanShift(bandwidth=2),
    'DBSCAN': DBSCAN(eps=1, min_samples=10),
}

plt.figure()
plt.suptitle('Iris data')
for row, model_name in enumerate(datasets.keys()):
    plt.subplot(4, 2, (row+1)*2-1)
    plt.title('Real clusters')
    scatter = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='tab20b')
    plt.title('Real clusters')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlabel('Length of sepals')
    plt.ylabel('Length of petals')

    model = datasets[model_name].fit(X_iris)
    plt.subplot(4, 2, (row+1)*2)
    plt.title(f'Cluster with {model_name}')
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=model.labels_, cmap='tab20b')
    plt.xlabel('Length of sepals')
    plt.ylabel('Length of petals')

plt.tight_layout()
plt.show()
