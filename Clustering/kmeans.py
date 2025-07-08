import ucimlrepo
import time

import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import silhouette_score

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
            
        # Randomly initialize centroids
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            self._update_centroids(X)
            
            # Check for convergence
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def _assign_clusters(self, X):
        # Calculate distances between points and centroids
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each point to nearest centroid
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X):
        # Update centroids as mean of assigned points
        for k in range(self.n_clusters):
            if np.sum(self.labels == k) > 0:  # Avoid empty clusters
                self.centroids[k] = np.mean(X[self.labels == k], axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)
    
    def compute_wcss(self, X):
        # Calculate Within-Cluster Sum of Squares
        wcss = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[k])**2)
        return wcss

# Fetch and process Iris dataset
if __name__ == "__main__":
    # Fetch dataset
    iris = fetch_ucirepo(id=352)
    
    # Data (as pandas dataframes)
    X = iris.data.features.to_numpy()  # Convert to numpy array
    
    # Lists to store metrics
    wcss_values = []
    silhouette_values = []

    iterations = 1
    
    # Test K-means for k from 2 to 10
    for k in range(2, 11):
        wcssss = 0
        silh = 0
        for iteration in range(iterations):
        # Create and fit K-means model
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
        
        # Calculate WCSS
            wcss = kmeans.compute_wcss(X)
            wcssss += wcss
        
        # Calculate Silhouette Score (if possible)
            if k < X.shape[0]:  # Silhouette score requires k < n_samples
                silhouette = silhouette_score(X, kmeans.labels)
                silh += silhouette
        
        wcss = wcssss / iterations
        silhouette = silh / iterations
        wcss_values.append(wcss)
        silhouette_values.append(silhouette)
        
        # Print results for this k
        print(f"\nResults for k={k}:")
        print(f"WCSS: {wcss:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}" if silhouette is not None else "Silhouette Score: N/A")
    
    # Print summary
    print("\nSummary of Metrics:")
    print("k\tWCSS\t\tSilhouette Score")
    print("-" * 40)
    for k, wcss, silhouette in zip(range(2, 11), wcss_values, silhouette_values):
        silhouette_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
        print(f"{k}\t{wcss:.4f}\t\t{silhouette_str}")