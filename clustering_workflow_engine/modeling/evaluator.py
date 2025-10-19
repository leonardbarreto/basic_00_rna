from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)

def evaluate_clustering(model, X):
    """
    Avalia métricas de clustering e retorna labels.
    """
    logger.info("Gerando previsões do clustering...")
    labels = model.fit_predict(X)

    logger.info("Calculando métricas de avaliação de clustering...")
    metrics = {}
    try:
        metrics["silhouette_score"] = silhouette_score(X, labels)
    except Exception:
        metrics["silhouette_score"] = None
    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics["calinski_harabasz_score"] = None
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
    except Exception:
        metrics["davies_bouldin_score"] = None

    return metrics, labels
