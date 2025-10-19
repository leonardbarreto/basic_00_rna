# clustering_workflow_engine/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

sns.set(style="whitegrid")

def plot_clusters(X, labels):
    """
    Plota clusters em 2D. Se X tiver mais de 2 features, reduz com PCA.
    Retorna matplotlib.figure.Figure.
    """
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X)
    else:
        X_plot = X

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        cluster_points = X_plot[labels == lbl]
        if lbl == -1:  # DBSCAN ruído
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c='k', label='Noise', marker='x')
        else:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {lbl}')
    ax.set_title("Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    return fig


def plot_silhouette_distribution(X, labels):
    """
    Plota a distribuição dos silhouette scores por ponto.
    Retorna matplotlib.figure.Figure.
    """
    # Silhouette só faz sentido se houver mais de 1 cluster
    if len(np.unique(labels)) <= 1:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Silhouette não aplicável", ha='center', va='center')
        ax.set_axis_off()
        return fig

    sil_scores = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(sil_scores, bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribuição dos Silhouette Scores")
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("Contagem")
    return fig


def plot_cluster_pca(X, labels):
    """
    Plota clusters usando PCA 2D, útil para datasets multidimensionais.
    Retorna matplotlib.figure.Figure.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    return plot_clusters(X_pca, labels)
