import numpy as np
from sklearn.preprocessing import StandardScaler
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


class KMeansClusterer:
    def __init__(self, k, max_iters=100, tolerance=1e-4, init_method='kmeans++', random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.init_method = init_method
        self.random_state = random_state
        
        # Results will be stored here after fitting
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        if self.init_method == 'random':
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))
        elif self.init_method == 'kmeans++':
            centroids = np.zeros((self.k, n_features))
            centroids[0] = X[np.random.randint(n_samples)]
            
            for i in range(1, self.k):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
                
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids[i] = X[j]
                        break
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)
        
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        return labels.astype(int)
    
    def _update_centroids(self, X, labels):
        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i] if self.centroids is not None else np.random.rand(n_features)
        return new_centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            labels = self._assign_clusters(X, self.centroids)
            new_centroids = self._update_centroids(X, labels)
            
            centroid_movement = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if centroid_movement < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                self.n_iter_ = iteration + 1
                break
        else:
            print(f"Reached maximum iterations ({self.max_iters})")
            self.n_iter_ = self.max_iters
        
        self.labels_ = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._calculate_inertia(X, self.labels_, self.centroids)
        
        return self
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    

def find_inertias(X, max_k=5):
    print(f"Finding optimal number of clusters (k=1 to {max_k})...")
    
    inertias = []
    
    for k in range(1, max_k + 1):
        print(f"Testing k={k}...")
        kmeans = KMeansClusterer(k=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias


def detect_optimal_k_elbow(inertias):
    inertias = np.array(inertias)
    n_points = len(inertias)
    
    x = np.arange(1, n_points + 1)
    y = inertias
    
    first_point = np.array([x[0], y[0]])
    last_point = np.array([x[-1], y[-1]])
    
    max_distance = 0
    optimal_k = 1
    
    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        line_vec = last_point - first_point
        point_vec = point - first_point
        line_len = np.linalg.norm(line_vec)
        
        if line_len > 0:
            line_unitvec = line_vec / line_len
            proj_length = np.dot(point_vec, line_unitvec)
            proj = proj_length * line_unitvec
            distance = np.linalg.norm(point_vec - proj)
            
            if distance > max_distance:
                max_distance = distance
                optimal_k = i + 1
    
    return optimal_k


def calculate_silhouette_coefficient(X, labels):
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1:
        return 0.0, np.zeros(n_samples)
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        point = X[i]
        own_cluster = labels[i]
        
        same_cluster_points = X[labels == own_cluster]
        if len(same_cluster_points) > 1:
            same_cluster_distances = [np.linalg.norm(point - other) for j, other in enumerate(same_cluster_points) if j != np.where(labels == own_cluster)[0].tolist().index(i)]
            a_i = np.mean(same_cluster_distances) if same_cluster_distances else 0
        else:
            a_i = 0
        
        b_i = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != own_cluster:
                other_cluster_points = X[labels == cluster_id]
                if len(other_cluster_points) > 0:
                    other_cluster_distances = [np.linalg.norm(point - other) for other in other_cluster_points]
                    mean_distance = np.mean(other_cluster_distances)
                    b_i = min(b_i, mean_distance)
        
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    silhouette_avg = np.mean(silhouette_scores)
    return silhouette_avg, silhouette_scores
