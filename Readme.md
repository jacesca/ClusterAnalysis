# Machine Learning - Cluster Analysis
Sample code to make cluster analysis.

Features:
- Partitional clustering
- Hierarchical Clustering: Agglomerative Clustering
- Density-based clustering: DBSCAN and MeanShift Clustering
- Pattern recognition
- Outliers detection / Anomaly detection
- Recommendation systems
- Real Data Example
- Cluster Evaluation

## Run ML model
```
python KMeansCluster.py
python HowKMeansWorks.py
python AglomerativeCluster.py
python HowAglomerativeClusterWorks.py
python MeanShiftCluster.py
python DBScanCluster.py
python RealDataCluster.py
python InternalEvaluation.py
python ExternalEvaluation.py
```

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/ClusterAnalysis.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Others
- Proyect in GitHub: https://github.com/jacesca/ClusterAnalysis
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Extra documentation
- [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
