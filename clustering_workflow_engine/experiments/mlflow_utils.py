import mlflow
from loguru import logger

def start_experiment(experiment_name: str):
    """
    Inicia ou retorna um experimento MLflow
    """
    mlflow.set_experiment(experiment_name)
    logger.info(f"Experimento MLflow iniciado: {experiment_name}")

def log_params(params: dict):
    """
    Loga parâmetros no MLflow
    """
    for key, value in params.items():
        mlflow.log_param(key, value)
    logger.info(f"Parâmetros logados: {params}")

def log_metrics(metrics: dict):
    """
    Loga métricas no MLflow
    """
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, value)
    logger.info(f"Métricas logadas: {metrics}")

def log_model(model):
    """
    Loga o modelo no MLflow
    """
    try:
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Modelo logado com sucesso")
    except Exception as e:
        logger.warning(f"Falha ao logar modelo: {e}")
