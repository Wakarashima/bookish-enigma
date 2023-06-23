# Import required libraries
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = load_iris()
X = data.data

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform data mining (K-Means clustering)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Print the cluster labels
print("Cluster labels:")
print(labels
