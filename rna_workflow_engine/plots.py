import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

sns.set(style="whitegrid")

def plot_clusters(X, labels):
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X)
    else:
        X_plot = X
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        cluster_points = X_plot[labels == lbl]
        if lbl == -1:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c='k', label='Noise', marker='x')
        else:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {lbl}')
    ax.set_title("Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    return fig

def plot_silhouette_distribution(X, labels):
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
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    return plot_clusters(X_pca, labels)

def plot_training_validation(train_losses, val_losses, fold: int = 1):
    """
    Plota as curvas de loss de treinamento e validação por fold.
    Ajusta eixo x para ter o mesmo tamanho.
    Retorna matplotlib.figure.Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Se train e val tiverem tamanhos diferentes, interpolamos val_losses
    epochs = np.arange(1, len(train_losses) + 1)
    
    if len(val_losses) != len(train_losses):
        val_losses_interp = np.interp(epochs, np.linspace(1, len(train_losses), len(val_losses)), val_losses)
    else:
        val_losses_interp = val_losses

    ax.plot(epochs, train_losses, label="Train Loss", marker='o')
    ax.plot(epochs, val_losses_interp, label="Validation Loss", marker='x')
    ax.set_title(f"Treinamento x Validação - Fold {fold}")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    return fig