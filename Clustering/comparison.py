import numpy as np
import matplotlib.pyplot as plt
import json
import os
from itertools import product
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

from k_means import (
    KMeansClusterer, 
    filter_numerical_features, 
    preprocess_data, 
    calculate_silhouette_coefficient as kmeans_silhouette,
    find_inertias,
    detect_optimal_k_elbow
)
from dbscan import (
    DBSCANClusterer,
    filter_numerical_features as dbscan_filter_features,
    preprocess_data as dbscan_preprocess,
    calculate_silhouette_coefficient as dbscan_silhouette,
    find_optimal_parameters
)
from ucimlrepo import fetch_ucirepo
from ssl_validation import disable_ssl_verification

# Colors for consistent visualization
KMEANS_COLOR = '#1f77b4'  # Blue
DBSCAN_COLOR = '#ff7f0e'  # Orange
DATASET_COLORS = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', "#3ec44a", '#bcbd22']

def load_dataset_by_id(dataset_id):
    try:
        print(f"Loading dataset ID: {dataset_id}")
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Filter numerical features
        filtered_dataset, removed_features = filter_numerical_features(dataset)
        
        if filtered_dataset.data.features.shape[1] == 0:
            print(f"⚠️ Dataset {dataset_id} has no numerical features, skipping...")
            return None, None, None, None
            
        # Extract data
        X = filtered_dataset.data.features.values
        y = filtered_dataset.data.targets.values.ravel()
        feature_names = list(filtered_dataset.data.features.columns)
        
        # Get dataset name from metadata
        dataset_name = getattr(dataset.metadata, 'name', f'Dataset_{dataset_id}')
        
        print(f"✅ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        if removed_features:
            print(f"   Removed {len(removed_features)} categorical features")
            
        return X, y, feature_names, dataset_name
        
    except Exception as e:
        print(f"❌ Error loading dataset {dataset_id}: {str(e)}")
        return None, None, None, None

def run_kmeans_analysis(X, dataset_name, max_k=5):
    print(f"Running K-means analysis for {dataset_name}...")
    
    # Find optimal k using elbow method
    inertias = find_inertias(X, max_k)
    
    # Detect optimal k
    optimal_k = detect_optimal_k_elbow(inertias)
    
    # Run K-means with optimal k
    kmeans = KMeansClusterer(k=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    if optimal_k > 1:
        silhouette_score, _ = kmeans_silhouette(X, labels)
    else:
        silhouette_score = 0.0
    
    # Calculate cluster sizes
    cluster_sizes = []
    for i in range(optimal_k):
        size = np.sum(labels == i)
        cluster_sizes.append(size)
    
    results = {
        'algorithm': 'K-Means',
        'dataset': dataset_name,
        'optimal_k': optimal_k,
        'labels': labels,
        'centroids': kmeans.centroids,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_score,
        'cluster_sizes': cluster_sizes,
        'inertias': inertias,
        'n_clusters': optimal_k,
        'n_noise': 0,  # K-means doesn't have noise points
        'noise_ratio': 0.0
    }
    
    print(f"   K-means: k={optimal_k}, silhouette={silhouette_score:.3f}")
    return results

def run_dbscan_analysis(X, dataset_name):
    print(f"Running DBSCAN analysis for {dataset_name}...")
    
    # Adjust max_combinations based on dataset size for efficiency
    n_samples = X.shape[0]
    if n_samples > 2000:
        max_combinations = 5  # Reduce combinations for large datasets
    elif n_samples > 1000:
        max_combinations = 10
    else:
        max_combinations = 15
    
    print(f"   Using {max_combinations} parameter combinations for {n_samples} samples")
    
    # Find optimal parameters
    param_results = find_optimal_parameters(X, max_combinations=max_combinations)
    best_params = param_results[0]
    
    # Run DBSCAN with optimal parameters
    dbscan = DBSCANClusterer(eps=best_params['eps'], min_samples=best_params['min_samples'])
    labels = dbscan.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_score = dbscan_silhouette(X, labels)
    
    # Calculate cluster sizes (excluding noise)
    cluster_sizes = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # Exclude noise
            size = np.sum(labels == label)
            cluster_sizes.append(size)
    
    results = {
        'algorithm': 'DBSCAN',
        'dataset': dataset_name,
        'optimal_eps': best_params['eps'],
        'optimal_min_samples': best_params['min_samples'],
        'labels': labels,
        'silhouette_score': silhouette_score,
        'cluster_sizes': cluster_sizes,
        'n_clusters': dbscan.n_clusters_,
        'n_noise': dbscan.n_noise_,
        'noise_ratio': best_params['noise_ratio'],
        'param_results': param_results
    }
    
    print(f"   DBSCAN: clusters={dbscan.n_clusters_}, noise_ratio={best_params['noise_ratio']:.3f}, silhouette={silhouette_score:.3f}")
    return results

def create_output_folder():
    """Create comparison folder for outputs."""
    folder_path = "comparison"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output folder: {folder_path}")
    else:
        print(f"Using existing output folder: {folder_path}")
    return folder_path

def plot_elbow_method_comparison(all_results, output_folder):
    """
    Plot elbow method curves for all datasets in the same graph.
    """
    plt.figure(figsize=(14, 10))
    
    for i, (dataset_name, results) in enumerate(all_results.items()):
        if 'inertias' in results['kmeans']:
            inertias = results['kmeans']['inertias']
            k_values = range(1, len(inertias) + 1)
            color = DATASET_COLORS[i % len(DATASET_COLORS)]
            
            plt.plot(k_values, inertias, 'o-', linewidth=2, markersize=6, 
                    label=f'{dataset_name}', color=color, alpha=0.8)
            
            # Mark optimal k
            optimal_k = results['kmeans']['optimal_k']
            plt.plot(optimal_k, inertias[optimal_k-1], 's', markersize=10, 
                    color=color, markeredgecolor='red', markeredgewidth=2)
            
            # Add text annotation for optimal k
            plt.annotate(f'k={optimal_k}', 
                        xy=(optimal_k, inertias[optimal_k-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title('Elbow Method Analysis - All Datasets (K-Means)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = os.path.join(output_folder, '1_elbow_method_all_datasets.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Elbow method comparison saved to: {filename}")
    plt.close()

def plot_silhouette_comparison_per_dataset(all_results, output_folder):
    """
    Plot silhouette analysis for each dataset comparing K-means and DBSCAN.
    """
    for dataset_name, results in all_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        kmeans_results = results['kmeans']
        dbscan_results = results['dbscan']
        
        # K-means silhouette by k
        if 'inertias' in kmeans_results:
            max_k = len(kmeans_results['inertias'])
            k_values = range(2, max_k + 1)  # Start from k=2 for silhouette
            silhouette_scores = []
            
            # Calculate silhouette scores for different k values
            X = results['X_scaled']
            for k in k_values:
                if k <= max_k:
                    kmeans_temp = KMeansClusterer(k=k, random_state=42)
                    labels_temp = kmeans_temp.fit_predict(X)
                    if k > 1:
                        sil_score, _ = kmeans_silhouette(X, labels_temp)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0.0)
            
            ax1.plot(k_values, silhouette_scores, 'o-', color=KMEANS_COLOR, 
                    linewidth=2, markersize=6, label='K-Means')
            ax1.axvline(x=kmeans_results['optimal_k'], color=KMEANS_COLOR, 
                       linestyle='--', alpha=0.7)
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title(f'K-Means Silhouette Analysis\n{dataset_name}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # DBSCAN parameter analysis
        if 'param_results' in dbscan_results:
            param_results = dbscan_results['param_results']
            eps_values = [r['eps'] for r in param_results]
            silhouette_scores = [r['silhouette_score'] for r in param_results]
            
            # Create scatter plot of eps vs silhouette score
            scatter = ax2.scatter(eps_values, silhouette_scores, c=silhouette_scores, 
                                cmap='viridis', s=60, alpha=0.7)
            
            # Mark optimal point
            optimal_eps = dbscan_results['optimal_eps']
            optimal_sil = dbscan_results['silhouette_score']
            ax2.scatter(optimal_eps, optimal_sil, color='red', s=100, 
                       marker='*', edgecolor='darkred', linewidth=2, 
                       label=f'Optimal (eps={optimal_eps:.3f})')
            
            ax2.set_xlabel('Eps Parameter')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title(f'DBSCAN Parameter Analysis\n{dataset_name}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.colorbar(scatter, ax=ax2, label='Silhouette Score')
        
        plt.tight_layout()
        filename = os.path.join(output_folder, f'2_silhouette_analysis_{dataset_name.lower().replace(" ", "_")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Silhouette analysis for {dataset_name} saved to: {filename}")
        plt.close()

def plot_cluster_size_comparison_per_dataset(all_results, output_folder):
    """
    Plot cluster size comparison for each dataset between K-means and DBSCAN.
    """
    for dataset_name, results in all_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        kmeans_results = results['kmeans']
        dbscan_results = results['dbscan']
        
        # K-means cluster sizes
        kmeans_sizes = kmeans_results['cluster_sizes']
        if kmeans_sizes:
            clusters_k = range(len(kmeans_sizes))
            bars1 = ax1.bar(clusters_k, kmeans_sizes, color=KMEANS_COLOR, alpha=0.7, 
                           label='K-Means Clusters')
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Number of Points')
            ax1.set_title(f'K-Means Cluster Sizes\n{dataset_name}\n(k={len(kmeans_sizes)})')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, size in zip(bars1, kmeans_sizes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{size}', ha='center', va='bottom', fontsize=9)
        
        # DBSCAN cluster sizes
        dbscan_sizes = dbscan_results['cluster_sizes']
        noise_count = dbscan_results['n_noise']
        
        if dbscan_sizes:
            clusters_d = range(len(dbscan_sizes))
            bars2 = ax2.bar(clusters_d, dbscan_sizes, color=DBSCAN_COLOR, alpha=0.7, 
                           label='DBSCAN Clusters')
            
            # Add noise bar if there are noise points
            if noise_count > 0:
                noise_bar = ax2.bar(len(dbscan_sizes), noise_count, color='red', alpha=0.7, 
                                   label='Noise Points')
                ax2.text(len(dbscan_sizes), noise_count + noise_count*0.01,
                        f'{noise_count}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xlabel('Cluster ID (+ Noise)')
            ax2.set_ylabel('Number of Points')
            ax2.set_title(f'DBSCAN Cluster Sizes\n{dataset_name}\n({len(dbscan_sizes)} clusters + {noise_count} noise)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add value labels on cluster bars
            for bar, size in zip(bars2, dbscan_sizes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{size}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = os.path.join(output_folder, f'3_cluster_sizes_{dataset_name.lower().replace(" ", "_")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Cluster size comparison for {dataset_name} saved to: {filename}")
        plt.close()

def plot_silhouette_score_by_parameters(all_results, output_folder):
    """
    Plot silhouette score by DBSCAN parameters for each dataset.
    """
    for dataset_name, results in all_results.items():
        if 'param_results' in results['dbscan']:
            param_results = results['dbscan']['param_results']
            
            # Extract parameter data
            eps_values = [r['eps'] for r in param_results]
            min_samples_values = [r['min_samples'] for r in param_results]
            silhouette_scores = [r['silhouette_score'] for r in param_results]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(eps_values, min_samples_values, c=silhouette_scores, 
                                cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Mark optimal point
            optimal_eps = results['dbscan']['optimal_eps']
            optimal_min_samples = results['dbscan']['optimal_min_samples']
            optimal_silhouette = results['dbscan']['silhouette_score']
            
            plt.scatter(optimal_eps, optimal_min_samples, color='red', s=200, 
                       marker='*', edgecolor='darkred', linewidth=2, 
                       label=f'Optimal (eps={optimal_eps:.3f}, min_samples={optimal_min_samples})')
            
            plt.xlabel('Eps Parameter', fontsize=12)
            plt.ylabel('Min Samples Parameter', fontsize=12)
            plt.title(f'DBSCAN: Silhouette Score by Parameters\n{dataset_name}', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Silhouette Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(output_folder, f'4_silhouette_by_params_{dataset_name.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Silhouette score by parameters for {dataset_name} saved to: {filename}")
            plt.close()

def plot_noise_ratio_by_parameters(all_results, output_folder):
    """
    Plot noise ratio by DBSCAN parameters for each dataset.
    """
    for dataset_name, results in all_results.items():
        if 'param_results' in results['dbscan']:
            param_results = results['dbscan']['param_results']
            
            # Extract parameter data
            eps_values = [r['eps'] for r in param_results]
            min_samples_values = [r['min_samples'] for r in param_results]
            noise_ratios = [r['noise_ratio'] for r in param_results]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(eps_values, min_samples_values, c=noise_ratios, 
                                cmap='Reds', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Mark optimal point
            optimal_eps = results['dbscan']['optimal_eps']
            optimal_min_samples = results['dbscan']['optimal_min_samples']
            optimal_noise_ratio = results['dbscan']['noise_ratio']
            
            plt.scatter(optimal_eps, optimal_min_samples, color='blue', s=200, 
                       marker='*', edgecolor='darkblue', linewidth=2, 
                       label=f'Optimal (eps={optimal_eps:.3f}, min_samples={optimal_min_samples})')
            
            plt.xlabel('Eps Parameter', fontsize=12)
            plt.ylabel('Min Samples Parameter', fontsize=12)
            plt.title(f'DBSCAN: Noise Ratio by Parameters\n{dataset_name}', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, label='Noise Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(output_folder, f'5_noise_ratio_by_params_{dataset_name.lower().replace(" ", "_")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Noise ratio by parameters for {dataset_name} saved to: {filename}")
            plt.close()

def plot_clustering_visualizations(all_results, output_folder):
    """
    Create side-by-side visualizations of K-means and DBSCAN clustering results for each dataset.
    Uses PCA for dimensionality reduction when datasets have more than 2 features.
    """
    for dataset_name, results in all_results.items():
        # Get the data and results
        X = results['X_scaled']
        kmeans_labels = results['kmeans']['labels']
        dbscan_labels = results['dbscan']['labels']
        
        # Apply PCA if data has more than 2 dimensions
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
            pca_info = f"PCA: {explained_variance[0]:.1%} + {explained_variance[1]:.1%} = {sum(explained_variance):.1%} variance"
        else:
            X_2d = X
            pca_info = "Original 2D data"
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # K-means visualization
        kmeans_unique_labels = np.unique(kmeans_labels)
        n_kmeans_clusters = len(kmeans_unique_labels)
        
        # Use a colormap for K-means
        colors_kmeans = plt.cm.Set1(np.linspace(0, 1, n_kmeans_clusters))
        
        for i, label in enumerate(kmeans_unique_labels):
            mask = kmeans_labels == label
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors_kmeans[i]], s=50, alpha=0.7, 
                       label=f'Cluster {label}', edgecolors='black', linewidth=0.5)
        
        # Plot K-means centroids if available and data is 2D or we can project them
        if X.shape[1] <= 2 and 'centroids' in results['kmeans']:
            centroids = results['kmeans']['centroids']
            ax1.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=200, linewidth=3, label='Centroids')
        elif X.shape[1] > 2 and 'centroids' in results['kmeans']:
            # Project centroids to 2D using the same PCA
            centroids_2d = pca.transform(results['kmeans']['centroids'])
            ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                       c='red', marker='x', s=200, linewidth=3, label='Centroids')
        
        ax1.set_title(f'K-Means Clustering\n{dataset_name}\n(k={results["kmeans"]["optimal_k"]}, silhouette={results["kmeans"]["silhouette_score"]:.3f})')
        ax1.set_xlabel('First Component')
        ax1.set_ylabel('Second Component')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # DBSCAN visualization
        dbscan_unique_labels = np.unique(dbscan_labels)
        n_dbscan_clusters = len(dbscan_unique_labels[dbscan_unique_labels != -1])  # Exclude noise
        
        # Use a colormap for DBSCAN clusters
        colors_dbscan = plt.cm.Set2(np.linspace(0, 1, max(n_dbscan_clusters, 1)))
        
        cluster_color_idx = 0  # Track color index for non-noise clusters
        for label in dbscan_unique_labels:
            mask = dbscan_labels == label
            if label == -1:  # Noise points
                ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c='red', s=50, alpha=0.6, marker='x', 
                           label='Noise', linewidth=1)
            else:  # Cluster points
                ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=[colors_dbscan[cluster_color_idx]], s=50, alpha=0.7, 
                           label=f'Cluster {label}', edgecolors='black', linewidth=0.5)
                cluster_color_idx += 1
        
        ax2.set_title(f'DBSCAN Clustering\n{dataset_name}\n(eps={results["dbscan"]["optimal_eps"]:.3f}, min_samples={results["dbscan"]["optimal_min_samples"]}, silhouette={results["dbscan"]["silhouette_score"]:.3f})')
        ax2.set_xlabel('First Component')
        ax2.set_ylabel('Second Component')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add PCA information if used
        fig.suptitle(f'Clustering Comparison: {dataset_name}\n{pca_info}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(output_folder, f'6_clustering_visualization_{dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Clustering visualization for {dataset_name} saved to: {filename}")
        plt.close()

def create_comparison_report(all_results, output_folder):
    """
    Create a comprehensive text report comparing K-means and DBSCAN results.
    """
    report_path = os.path.join(output_folder, 'comparison.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE COMPARISON: K-MEANS vs DBSCAN CLUSTERING\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        for dataset_name, results in all_results.items():
            X_shape = results['X_scaled'].shape
            f.write(f"{dataset_name}:\n")
            f.write(f"  - Samples: {X_shape[0]}\n")
            f.write(f"  - Features: {X_shape[1]}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("1. CLUSTER SIZE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"{dataset_name.upper()}\n")
            f.write("-" * len(dataset_name) + "\n")
            
            kmeans_results = results['kmeans']
            dbscan_results = results['dbscan']
            
            # K-means cluster sizes
            f.write(f"K-Means (k={kmeans_results['optimal_k']}):\n")
            for i, size in enumerate(kmeans_results['cluster_sizes']):
                percentage = (size / sum(kmeans_results['cluster_sizes'])) * 100
                f.write(f"  Cluster {i}: {size} points ({percentage:.1f}%)\n")
            
            f.write(f"\nDBSCAN (eps={dbscan_results['optimal_eps']:.3f}, min_samples={dbscan_results['optimal_min_samples']}):\n")
            total_non_noise = sum(dbscan_results['cluster_sizes'])
            total_points = total_non_noise + dbscan_results['n_noise']
            
            for i, size in enumerate(dbscan_results['cluster_sizes']):
                percentage = (size / total_points) * 100
                f.write(f"  Cluster {i}: {size} points ({percentage:.1f}%)\n")
            
            if dbscan_results['n_noise'] > 0:
                noise_percentage = (dbscan_results['n_noise'] / total_points) * 100
                f.write(f"  Noise: {dbscan_results['n_noise']} points ({noise_percentage:.1f}%)\n")
            
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("2. SILHOUETTE SCORE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("Algorithm Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Dataset':<20} {'K-Means':<15} {'DBSCAN':<15} {'Winner':<10}\n")
        f.write("-" * 65 + "\n")
        
        kmeans_wins = 0
        dbscan_wins = 0
        
        for dataset_name, results in all_results.items():
            kmeans_sil = results['kmeans']['silhouette_score']
            dbscan_sil = results['dbscan']['silhouette_score']
            
            if kmeans_sil > dbscan_sil:
                winner = "K-Means"
                kmeans_wins += 1
            elif dbscan_sil > kmeans_sil:
                winner = "DBSCAN"
                dbscan_wins += 1
            else:
                winner = "Tie"
            
            f.write(f"{dataset_name:<20} {kmeans_sil:<15.4f} {dbscan_sil:<15.4f} {winner:<10}\n")
        
        f.write("-" * 65 + "\n")
        f.write(f"Total Wins: K-Means={kmeans_wins}, DBSCAN={dbscan_wins}\n\n")
        
        # Detailed analysis
        f.write("Detailed Analysis:\n")
        f.write("-" * 20 + "\n")
        for dataset_name, results in all_results.items():
            f.write(f"\n{dataset_name}:\n")
            
            kmeans_results = results['kmeans']
            dbscan_results = results['dbscan']
            
            f.write(f"  K-Means:\n")
            f.write(f"    - Optimal k: {kmeans_results['optimal_k']}\n")
            f.write(f"    - Silhouette Score: {kmeans_results['silhouette_score']:.4f}\n")
            f.write(f"    - Inertia: {kmeans_results['inertia']:.4f}\n")
            f.write(f"    - All points clustered (no noise)\n")
            
            f.write(f"  DBSCAN:\n")
            f.write(f"    - Optimal eps: {dbscan_results['optimal_eps']:.4f}\n")
            f.write(f"    - Optimal min_samples: {dbscan_results['optimal_min_samples']}\n")
            f.write(f"    - Silhouette Score: {dbscan_results['silhouette_score']:.4f}\n")
            f.write(f"    - Number of clusters: {dbscan_results['n_clusters']}\n")
            f.write(f"    - Noise ratio: {dbscan_results['noise_ratio']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("3. ALGORITHM CHARACTERISTICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("K-Means Characteristics:\n")
        f.write("- Requires pre-specification of number of clusters (k)\n")
        f.write("- Assumes spherical clusters of similar size\n")
        f.write("- Every point is assigned to a cluster (no noise concept)\n")
        f.write("- Sensitive to initialization (uses k-means++ here)\n")
        f.write("- Computationally efficient\n")
        f.write("- Works well with well-separated, spherical clusters\n\n")
        
        f.write("DBSCAN Characteristics:\n")
        f.write("- Automatically determines number of clusters\n")
        f.write("- Can find clusters of arbitrary shape\n")
        f.write("- Identifies noise/outlier points\n")
        f.write("- Requires tuning of eps and min_samples parameters\n")
        f.write("- More robust to outliers\n")
        f.write("- Works well with non-spherical clusters and varying densities\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        best_overall_kmeans = sum(results['kmeans']['silhouette_score'] for results in all_results.values()) / len(all_results)
        best_overall_dbscan = sum(results['dbscan']['silhouette_score'] for results in all_results.values()) / len(all_results)
        
        f.write(f"Average Silhouette Scores:\n")
        f.write(f"- K-Means: {best_overall_kmeans:.4f}\n")
        f.write(f"- DBSCAN: {best_overall_dbscan:.4f}\n\n")
        
        if best_overall_kmeans > best_overall_dbscan:
            f.write("Overall Recommendation: K-Means performs better on average across all datasets.\n")
        else:
            f.write("Overall Recommendation: DBSCAN performs better on average across all datasets.\n")
        
        f.write("\nDataset-specific recommendations:\n")
        for dataset_name, results in all_results.items():
            kmeans_sil = results['kmeans']['silhouette_score']
            dbscan_sil = results['dbscan']['silhouette_score']
            
            if kmeans_sil > dbscan_sil:
                f.write(f"- {dataset_name}: Use K-Means (silhouette: {kmeans_sil:.4f} vs {dbscan_sil:.4f})\n")
            else:
                f.write(f"- {dataset_name}: Use DBSCAN (silhouette: {dbscan_sil:.4f} vs {kmeans_sil:.4f})\n")
    
    print(f"Comparison report saved to: {report_path}")

def main():
    print("="*80)
    print("COMPREHENSIVE CLUSTERING COMPARISON: K-MEANS vs DBSCAN")
    print("="*80)
    
    # Disable SSL verification
    disable_ssl_verification()
    
    # Create output folder
    output_folder = create_output_folder()
    
    # Load datasets configuration
    try:
        with open('datasets.json', 'r') as f:
            datasets_config = json.load(f)
    except FileNotFoundError:
        print("❌ datasets.json file not found!")
        return
    
    print(f"\nLoading {len(datasets_config)} datasets for comparison...")
    
    all_results = {}
    
    # Process each dataset
    for dataset_name, dataset_id in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name} (ID: {dataset_id})")
        print(f"{'='*60}")
        
        # Load dataset
        X, y, feature_names, actual_name = load_dataset_by_id(dataset_id)
        
        if X is None:
            print(f"Skipping {dataset_name} due to loading error")
            continue
        
        # Use actual dataset name if available
        display_name = actual_name if actual_name else dataset_name
        
        # Preprocess data
        print("Preprocessing data...")
        X_scaled, scaler, valid_indices = preprocess_data(X)
        
        if X_scaled.shape[0] < 10:
            print(f"⚠️ Dataset {display_name} has too few samples after preprocessing, skipping...")
            continue
        
        # Run K-means analysis
        kmeans_results = run_kmeans_analysis(X_scaled, display_name)
        
        # Run DBSCAN analysis
        dbscan_results = run_dbscan_analysis(X_scaled, display_name)
        
        # Store results
        all_results[display_name] = {
            'kmeans': kmeans_results,
            'dbscan': dbscan_results,
            'X_scaled': X_scaled,
            'original_shape': X.shape,
            'feature_names': feature_names
        }
    
    if not all_results:
        print("❌ No datasets were successfully processed!")
        return
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Generate all comparison plots
    print("\n1. Creating elbow method comparison...")
    plot_elbow_method_comparison(all_results, output_folder)
    
    print("\n2. Creating silhouette analysis comparisons...")
    plot_silhouette_comparison_per_dataset(all_results, output_folder)
    
    print("\n3. Creating cluster size comparisons...")
    plot_cluster_size_comparison_per_dataset(all_results, output_folder)
    
    # print("\n4. Creating silhouette score by parameters plots...")
    # plot_silhouette_score_by_parameters(all_results, output_folder)
    
    # print("\n5. Creating noise ratio by parameters plots...")
    # plot_noise_ratio_by_parameters(all_results, output_folder)
    
    print("\n6. Creating clustering visualizations...")
    plot_clustering_visualizations(all_results, output_folder)
    
    print("\n7. Creating comprehensive text report...")
    create_comparison_report(all_results, output_folder)
    
    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to '{output_folder}' folder:")
    print(f"- 1_elbow_method_all_datasets.png: Elbow curves for all datasets")
    print(f"- 2_silhouette_analysis_*.png: Silhouette comparison per dataset")
    print(f"- 3_cluster_sizes_*.png: Cluster size comparison per dataset")
    # print(f"- 4_silhouette_by_params_*.png: DBSCAN silhouette by parameters")
    # print(f"- 5_noise_ratio_by_params_*.png: DBSCAN noise ratio by parameters")
    print(f"- 6_clustering_visualization_*.png: Clustering results visualization")
    print(f"- comparison.txt: Comprehensive comparison report")
    
    print(f"\nProcessed {len(all_results)} datasets successfully:")
    for dataset_name in all_results.keys():
        print(f"  ✅ {dataset_name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
