"""
Part B: Anuran Calls (MFCCs) clustering pipeline as a script.

Usage:
python partB_KMeans.py
    --data data/Frogs_MFCCs.csv
    --random_state 42
    --pca_components 10
    --k 4
    --dbscan_eps 0.5
    --dbscan_min_samples 5

Outputs:
  - outputs/part_B/figures/*.png
  - outputs/part_B/metrics/*.json
  - outputs/part_B/models/*.joblib
  - outputs/part_B/artifacts/*.csv
"""
import os
import sys
import json
import argparse
import time
import platform

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from joblib import dump

# ========= Fixed defaults =========
DEFAULT_DATA_PATH = "/data/Frogs_MFCCs.csv"
DEFAULT_OUT_DIR = "outputs/part_B"

# ========= Utilities =========
def ensure_dirs(base_out):
    dirs = {
        "fig": os.path.join(base_out, "figures"),
        "models": os.path.join(base_out, "models"),
        "metrics": os.path.join(base_out, "metrics"),
        "artifacts": os.path.join(base_out, "artifacts"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def save_fig(dirs, name):
    path = os.path.join(dirs["fig"], f"{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved fig] {path}")

def save_csv(dirs, df, name):
    path = os.path.join(dirs["metrics"], f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[saved csv] {path}")

def save_json(dirs, obj, name):
    path = os.path.join(dirs["metrics"], f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved json] {path}")

def save_artifact(dirs, df_or_arr, name):
    path = os.path.join(dirs["artifacts"], f"{name}.csv")
    if isinstance(df_or_arr, pd.DataFrame):
        df_or_arr.to_csv(path, index=False)
    else:
        pd.DataFrame(df_or_arr).to_csv(path, index=False)
    print(f"[saved artifact] {path}")

def save_model(dirs, model, name):
    path = os.path.join(dirs["models"], f"{name}.joblib")
    dump(model, path)
    print(f"[saved model] {path}")

def log_run_metadata(dirs, args, np, pd, seaborn, plt):
    try:
        import sklearn
        import matplotlib
    except Exception:
        sklearn = None
        matplotlib = None
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seaborn": seaborn.__version__,
        "matplotlib": getattr(matplotlib, "__version__", "unknown"),
        "sklearn": getattr(sklearn, "__version__", "unknown"),
        "args": vars(args),
        "data_path": args.data,
        "out_dir": DEFAULT_OUT_DIR,
    }
    save_json(dirs, meta, "run_metadata")

# ========= Feature importance =========
def analyze_feature_importance(dirs, kmeans, pca, base_feature_names):
    centers_pca = kmeans.cluster_centers_
    centers_original = pca.inverse_transform(centers_pca)

    feature_importance = np.var(centers_original, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance in Clustering')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (Variance)')
    save_fig(dirs, "feature_importance_clustering")

    names = list(base_feature_names)
    extra_count = len(feature_importance) - len(base_feature_names)
    names.extend([f"poly_extra_{i+1}" for i in range(extra_count)])
    imp_df = pd.DataFrame({"feature": names, "importance_variance": feature_importance})
    save_csv(dirs, imp_df, "feature_importance_vector")

    return feature_importance

# ========= Pipeline =========
def run_pipeline(args):
    np.random.seed(args.random_state)

    # Outputs
    dirs = ensure_dirs(DEFAULT_OUT_DIR)
    log_run_metadata(dirs, args, np, pd, sns, plt)

    # Load data
    data = pd.read_csv(args.data)

    print("Dataset Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nBasic Statistics:")
    print(data.describe())

    # EDA: distributions (first 3 numeric)
    plt.figure(figsize=(15, 5))
    num_cols = data.select_dtypes(include=np.number).columns[:3]
    for i, column in enumerate(num_cols):
        plt.subplot(1, 3, i+1)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    save_fig(dirs, "distributions_first_3_numeric")

    # Correlation
    corr_matrix = data.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    save_fig(dirs, "feature_correlation_matrix")

    # Remove highly correlated features (threshold = 0.95)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_df = data.drop(columns=to_drop).select_dtypes(include=np.number)
    X = X_df.values
   
    save_csv(dirs, pd.DataFrame({"dropped_feature": to_drop}), "high_corr_dropped_features")

    # RecordID if present
    if "RecordID" in data.columns:
        record_ids = data["RecordID"].values
    else:
        record_ids = np.arange(len(data))

    # Feature engineering: Polynomial on first 5 numeric cols
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[:, :5])
    X = np.hstack((X, X_poly[:, X.shape[1]:]))
    save_model(dirs, poly, "poly_features_degree2")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_model(dirs, scaler, "standard_scaler")

    # PCA
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)
    save_model(dirs, pca, f"pca_{args.pca_components}_components")

    pca_var = pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_)
    })
    save_csv(dirs, pca_var, "pca_explained_variance")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(alpha=0.3)
    save_fig(dirs, "pca_cumulative_explained_variance")

    emb_df = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
    emb_df["RecordID"] = record_ids
    save_artifact(dirs, emb_df, "pca_embedding_pc1_pc3")

    # Elbow method (1..10)
    wcss = []
    ks = list(range(1, 11))
    for k in ks:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                    n_init=10, random_state=args.random_state)
        km.fit(X_pca)
        wcss.append(km.inertia_)
    elbow_df = pd.DataFrame({"k": ks, "wcss": wcss})
    save_csv(dirs, elbow_df, "elbow_wcss")
    plt.figure(figsize=(8, 6))
    plt.plot(ks, wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(alpha=0.3)
    save_fig(dirs, "elbow_method")

    # Init comparison
    kbest_clusters = args.k
    init_methods = ['random', 'k-means++']
    init_scores = {}
    for init in init_methods:
        kmeans_init = KMeans(n_clusters=kbest_clusters, init=init, max_iter=300,
                             n_init=10, random_state=args.random_state)
        labels_init = kmeans_init.fit_predict(X_pca)
        init_scores[init] = float(silhouette_score(X_pca, labels_init))
    print("\nK-means initialization comparison:", init_scores)
    save_json(dirs, init_scores, "kmeans_init_comparison")

    # Final KMeans
    kmeans = KMeans(n_clusters=kbest_clusters, init='k-means++', max_iter=300,
                    n_init=10, random_state=args.random_state)
    kmeans_labels = kmeans.fit_predict(X_pca)
    save_model(dirs, kmeans, f"kmeans_k{kbest_clusters}_kpp")
    save_artifact(dirs, pd.DataFrame({"RecordID": record_ids, "kmeans_label": kmeans_labels}),
                  f"kmeans_labels_k{kbest_clusters}")

    # Feature contribution
    feature_importance = analyze_feature_importance(dirs, kmeans, pca, X_df.columns)
    print("\nTop 5 most important features:", np.argsort(feature_importance)[-5:])

    # KMeans metrics
    silhouette_avg = silhouette_score(X_pca, kmeans_labels)
    db_index = davies_bouldin_score(X_pca, kmeans_labels)
    ch_index = calinski_harabasz_score(X_pca, kmeans_labels)
    print(f'\nK-Means: Silhouette Score={silhouette_avg:.3f}, Davies-Bouldin Index={db_index:.3f}, Calinski-Harabasz Index={ch_index:.3f}')
    kmeans_metrics = pd.DataFrame([{
        "method": "KMeans",
        "k": kbest_clusters,
        "silhouette": silhouette_avg,
        "davies_bouldin": db_index,
        "calinski_harabasz": ch_index
    }])
    save_csv(dirs, kmeans_metrics, "kmeans_metrics")

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=kbest_clusters)
    hierarchical_labels = hierarchical.fit_predict(X_pca)
    h_silhouette = silhouette_score(X_pca, hierarchical_labels)
    h_db = davies_bouldin_score(X_pca, hierarchical_labels)
    h_ch = calinski_harabasz_score(X_pca, hierarchical_labels)
    print(f'Hierarchical Clustering: Silhouette={h_silhouette:.3f}, Davies-Bouldin={h_db:.3f}, Calinski-Harabasz={h_ch:.3f}')
    hier_metrics = pd.DataFrame([{
        "method": "Agglomerative",
        "k": kbest_clusters,
        "silhouette": h_silhouette,
        "davies_bouldin": h_db,
        "calinski_harabasz": h_ch
    }])
    save_csv(dirs, hier_metrics, "hierarchical_metrics")
    save_artifact(dirs, pd.DataFrame({"RecordID": record_ids, "hierarchical_label": hierarchical_labels}),
                  f"hierarchical_labels_k{kbest_clusters}")

    # DBSCAN
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_pca)
    save_model(dirs, dbscan, f"dbscan_eps{str(args.dbscan_eps).replace('.','p')}_min{args.dbscan_min_samples}")

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_metrics_rows = []
    if len(set(dbscan_labels[dbscan_labels >= 0])) > 1:
        db_silhouette = silhouette_score(X_pca, dbscan_labels)
        db_db = davies_bouldin_score(X_pca, dbscan_labels)
        db_ch = calinski_harabasz_score(X_pca, dbscan_labels)
        print(f'DBSCAN: Silhouette={db_silhouette:.3f}, Davies-Bouldin={db_db:.3f}, Calinski-Harabasz={db_ch:.3f}')
        dbscan_metrics_rows.append({
            "method": "DBSCAN",
            "clusters": n_clusters_dbscan,
            "silhouette": db_silhouette,
            "davies_bouldin": db_db,
            "calinski_harabasz": db_ch
        })
    else:
        dbscan_metrics_rows.append({
            "method": "DBSCAN",
            "clusters": n_clusters_dbscan,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan
        })
    print(f'Number of clusters found by DBSCAN: {n_clusters_dbscan}')
    save_csv(dirs, pd.DataFrame(dbscan_metrics_rows), "dbscan_metrics")
    save_artifact(dirs, pd.DataFrame({"RecordID": record_ids, "dbscan_label": dbscan_labels}),
                  "dbscan_labels")

    # Visualization comparisons
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=10)
    plt.title('K-Means Clustering')
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', s=10)
    plt.title('Hierarchical Clustering')
    plt.subplot(1, 3, 3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', s=10)
    plt.title('DBSCAN Clustering')
    plt.tight_layout()
    save_fig(dirs, "clustering_comparison_pc12")

    # Aggregate metrics
    all_metrics = pd.concat([kmeans_metrics, hier_metrics, pd.DataFrame(dbscan_metrics_rows)],
                            ignore_index=True)
    save_csv(dirs, all_metrics, "all_clustering_metrics")

    print("\nDone. All figures, models, metrics, and artifacts saved under outputs/.")

# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser(description="Part B: Anuran Calls clustering pipeline runner ")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to Frogs_MFCCs.csv")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument("--pca_components", type=int, default=10, help="Number of PCA components")
    p.add_argument("--k", type=int, default=4, help="K for KMeans and Agglomerative")
    p.add_argument("--dbscan_eps", type=float, default=0.5, help="DBSCAN eps")
    p.add_argument("--dbscan_min_samples", type=int, default=5, help="DBSCAN min_samples")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
