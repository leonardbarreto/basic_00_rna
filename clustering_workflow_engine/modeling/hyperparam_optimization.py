import optuna
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def optimize_clustering(model_type: str, X, n_trials: int = 20):
    """
    Otimiza hiperparâmetros de modelos de clustering usando Optuna.

    Args:
        model_type: "kmeans" ou "dbscan"
        X: array-like, features para clustering
        n_trials: número de iterações do Optuna

    Returns:
        best_params: dicionário com os melhores parâmetros encontrados
    """
    def objective(trial):
        if model_type.lower() == "kmeans":
            n_clusters = trial.suggest_int("n_clusters", 2, 10)
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif model_type.lower() == "dbscan":
            eps = trial.suggest_float("eps", 0.1, 1.5)
            min_samples = trial.suggest_int("min_samples", 2, 10)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            raise ValueError(f"Modelo '{model_type}' não suportado")

        labels = model.fit_predict(X)
        # Se todos os pontos caírem em um único cluster, penaliza
        if len(set(labels)) <= 1:
            return -1.0
        return silhouette_score(X, labels)

    study = optuna.create_study(direction="maximize", study_name=f"{model_type}_optimization")
    print("Otimizing clustering hyperparameters...")  # mensagem inicial
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)  # sem barra de progresso
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")
    return best_params
