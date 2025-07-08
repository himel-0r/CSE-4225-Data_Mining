import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def filter_numerical_features(dataset):
    print("Categorical features...")
    
    features_df = dataset.data.features
    
    categorical_features = []
    numerical_features = []
    
    for col in features_df.columns:
        dtype = features_df[col].dtype
        
        if dtype == 'object' or dtype == 'string' or dtype == 'bool':
            categorical_features.append(col)
        elif hasattr(dtype, 'name') and 'category' in str(dtype):
            categorical_features.append(col)
        else:
            numerical_features.append(col)
            
    if categorical_features:
        print(f"\\n Categorical features detected and will be removed:")
        for feature in categorical_features:
            print(f"   - {feature}")
        print(f"\\n Keeping {len(numerical_features)} numerical features for clustering:")
        for feature in numerical_features:
            print(f"   - {feature}")
            
        filtered_features_df = features_df[numerical_features]
        
        filtered_dataset = type(dataset)()
        filtered_dataset.data = type(dataset.data)()
        filtered_dataset.data.features = filtered_features_df
        filtered_dataset.data.targets = dataset.data.targets
        
        return filtered_dataset, categorical_features
    print("All features are numerical. Proceeding with clustering...")
    return dataset, []


def preprocess_data(X):
    print("Preprocessing data...")
    
    if np.isnan(X).any():
        print("   Missing values detected. Removing rows with NaN values...")
        valid_indices = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_indices]
        print(f"   Removed {np.sum(~valid_indices)} rows with missing values")
        print(f"   Remaining samples: {X_clean.shape[0]}")
    else:
        print("   No missing values found in the dataset.")
        X_clean = X
        valid_indices = np.ones(X.shape[0], dtype=bool)
        
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    print("Data preprocessing completed.")
    
    return X_scaled, scaler, valid_indices


class DBSCANClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
        
    def _get_neighbors(self, X, point_idx):
        neighbors = []
        point = X[point_idx]
        
        for i, other_point in enumerate(X):
            distance = np.linalg.norm(point - other_point)
            if distance <= self.eps:
                neighbors.append(i)    
        return neighbors
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited, labels):
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_samples:
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)
            
            if labels[neighbor_idx] == -1:  # -1 means unassigned
                labels[neighbor_idx] = cluster_id
            
            i += 1
            
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        labels = np.full(n_samples, -1)
        visited = set()
        cluster_id = 0
        core_samples = []
        
        for point_idx in range(n_samples):
            if point_idx in visited:
                continue
            
            visited.add(point_idx)
            
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) >= self.min_samples:
                core_samples.append(point_idx)
                self._expand_cluster(X, point_idx, neighbors, cluster_id, visited, labels)
                cluster_id += 1
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise_ = list(labels).count(-1)
        
        return labels
    
    def fit(self, X):
        self.fit_predict(X)
        return self
    

def calculate_silhouette_coefficient(X, labels):
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) == 0:
        return 0.0
    
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]
    
    n_clusters = len(np.unique(labels_clean))
    if n_clusters <= 1:
        return 0.0
    
    n_samples = X_clean.shape[0]
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        point = X_clean[i]
        own_cluster = labels_clean[i]
        
        same_cluster_points = X_clean[labels_clean == own_cluster]
        if len(same_cluster_points) > 1:
            same_cluster_distances = []
            for j, other in enumerate(same_cluster_points):
                original_indices = np.where(labels_clean == own_cluster)[0]
                if original_indices[j] != i:
                    same_cluster_distances.append(np.linalg.norm(point - other))
            a_i = np.mean(same_cluster_distances) if same_cluster_distances else 0
        else:
            a_i = 0
        
        b_i = float('inf')
        for cluster_id in np.unique(labels_clean):
            if cluster_id != own_cluster:
                other_cluster_points = X_clean[labels_clean == cluster_id]
                if len(other_cluster_points) > 0:
                    other_cluster_distances = [np.linalg.norm(point - other) for other in other_cluster_points]
                    mean_distance = np.mean(other_cluster_distances)
                    b_i = min(b_i, mean_distance)
        
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)


def find_optimal_parameters(X, eps_range=None, min_samples_range=None, max_combinations=50):
    print("Finding optimal DBSCAN parameters...")
    
    if eps_range is None:
        print("  - Computing k-distance graph for eps range selection...")
        k = min(10, max(4, int(np.log(len(X)))))
        distances = []
        for point in X:
            dists = [np.linalg.norm(point - other) for other in X]
            dists.sort()
            distances.append(dists[k])
        
        distances.sort()
        eps_min = distances[len(distances) // 4]
        eps_max = distances[3 * len(distances) // 4]
        eps_range = np.linspace(eps_min, eps_max, 8)
        print(f"  - eps range: {eps_min:.3f} to {eps_max:.3f}")
    
    if min_samples_range is None:
        n_features = X.shape[1]
        min_samples_min = max(2, n_features // 2)
        min_samples_max = min(2 * n_features, max(6, X.shape[0] // 20))
        min_samples_range = range(min_samples_min, min_samples_max + 1)
        print(f"  - min_samples range: {min_samples_min} to {min_samples_max}")
    
    all_combinations = list(product(eps_range, min_samples_range))
    if len(all_combinations) > max_combinations:
        step = len(all_combinations) // max_combinations
        combinations = all_combinations[::step][:max_combinations]
    else:
        combinations = all_combinations
    
    results = []
    
    for i, (eps, min_samples) in enumerate(combinations):
        print(f"Testing combination {i+1}/{len(combinations)}: eps={eps:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCANClusterer(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = dbscan.n_clusters_
        n_noise = dbscan.n_noise_
        noise_ratio = n_noise / len(X) if len(X) > 0 else 1.0
        
        silhouette_score = calculate_silhouette_coefficient(X, labels)
        
        if n_clusters == 0:
            combined_score = 0
        else:
            combined_score = silhouette_score * (1 - noise_ratio * 0.5)
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': silhouette_score,
            'combined_score': combined_score
        })
    
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results
