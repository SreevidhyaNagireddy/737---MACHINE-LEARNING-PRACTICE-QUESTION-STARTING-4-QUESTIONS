import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# --- 1. Generate Synthetic Data ---
# We create 4 distinct "blobs" of data for clustering
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)

# --- 2. Apply Clustering Algorithms ---

# K-means
# We specify n_clusters=4 because we know our data has 4 blobs
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Gaussian Mixture Model (GMM)
# We specify n_components=4 (number of Gaussian distributions)
gmm = GaussianMixture(n_components=4, random_state=0)
y_gmm = gmm.fit_predict(X)

# Hierarchical (Agglomerative) Clustering
# We specify n_clusters=4, meaning we want to stop merging
# when we are left with 4 clusters
agg = AgglomerativeClustering(n_clusters=4)
y_agg = agg.fit_predict(X)

# --- 3. Plot the Results ---
plt.figure(figsize=(12, 12))

# Plot 1: Original Data (Ground Truth)
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=40, cmap='viridis')
plt.title("Original Data (Ground Truth)")

# Plot 2: K-means
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, cmap='viridis')
# Plot the K-means centroids
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("K-means Clustering")
plt.legend()

# Plot 3: Gaussian Mixture Model (GMM)
plt.subplot(2, 2, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis')
plt.title("Gaussian Mixture Model (GMM)")

# Plot 4: Hierarchical Clustering
plt.subplot(2, 2, 4)
plt.scatter(X[:, 0], X[:, 1], c=y_agg, s=40, cmap='viridis')
plt.title("Hierarchical Clustering")

plt.tight_layout()
plt.show()