# ai_workflow_engine/modeling/trainer.py
from sklearn.model_selection import ParameterGrid
from loguru import logger
from clustering_workflow_engine.modeling.evaluator import evaluate_clustering
from clustering_workflow_engine.experiments.mlflow_utils import log_params, log_metrics, log_model

def train_model_clustering(model_class, X, param_grid=None, mlflow_run=True):
    """
    Treina modelo de clustering testando diferentes combinações de parâmetros.
    
    Parâmetros
    ----------
    model_class : sklearn.cluster estimator
        Classe do modelo de clustering (ex: KMeans, DBSCAN).
    X : pd.DataFrame ou np.ndarray
        Dados de entrada.
    param_grid : dict
        Dicionário de parâmetros para teste (ex: {'n_clusters':[2,3,4]}).
    mlflow_run : bool
        Se True, loga parâmetros, métricas e modelo no MLflow.

    Retorna
    -------
    tuple
        Melhor modelo treinado, melhores métricas, labels gerados.
    """
    best_score = -1
    best_model = None
    best_params = None

    if param_grid is None:
        param_grid = {}

    logger.info("Iniciando grid search manual de parâmetros de clustering...")
    for params in ParameterGrid(param_grid):
        model = model_class(**params)
        metrics = evaluate_clustering(model, X)
        score = metrics["silhouette_score"]  # critério principal

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    logger.info(f"Melhor conjunto de parâmetros: {best_params}")
    if mlflow_run:
        log_params(best_params)
        log_metrics(evaluate_clustering(best_model, X))
        log_model(best_model)

    return best_model, evaluate_clustering(best_model, X)
