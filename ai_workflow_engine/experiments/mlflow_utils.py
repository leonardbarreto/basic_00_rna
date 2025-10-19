import mlflow
import mlflow.sklearn
from loguru import logger

def start_experiment(experiment_name: str):
    """
    Seleciona o experimento MLflow.
    """
    mlflow.set_experiment(experiment_name)
    logger.info(f"Experimento MLflow '{experiment_name}' selecionado")

def log_params(params: dict):
    """
    Loga parâmetros (hiperparâmetros) no MLflow.
    """
    for k, v in params.items():
        mlflow.log_param(k, v)
    logger.info("Hiperparâmetros logados no MLflow")

def log_metrics(metrics: dict):
    """
    Loga métricas no MLflow.
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    logger.info("Métricas logadas no MLflow")

def log_model(model, artifact_path="model"):
    """
    Loga o modelo treinado no MLflow.
    artifact_path: subpasta dentro do run onde o modelo será salvo.
    """
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    logger.info(f"Modelo salvo no MLflow em '{artifact_path}'")
