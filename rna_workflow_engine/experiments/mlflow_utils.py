import mlflow
from loguru import logger
import os

def start_experiment(experiment_name: str):
    """Inicia ou retorna um experimento MLflow"""
    mlflow.set_experiment(experiment_name)
    logger.info(f"Experimento MLflow iniciado: {experiment_name}")

def log_params(params: dict):
    """Loga parâmetros no MLflow"""
    for key, value in params.items():
        mlflow.log_param(key, value)
    logger.info(f"Parâmetros logados: {params}")

def log_metrics(metrics: dict):
    """Loga métricas no MLflow"""
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, value)
    logger.info(f"Métricas logadas: {metrics}")

def log_model(model):
    """Loga o modelo no MLflow"""
    try:
        mlflow.pytorch.log_model(model, artifact_path="model")
        logger.info("Modelo logado com sucesso")
    except Exception as e:
        logger.warning(f"Falha ao logar modelo: {e}")

import io

import mlflow
from matplotlib.figure import Figure
import tempfile

def log_figure(fig: Figure, artifact_name: str):
    """
    Loga uma figura do Matplotlib no MLflow como artefato.
    Salva temporariamente no disco antes do upload.
    """
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        tmp_path = tmpfile.name
        fig.savefig(tmp_path, format='png', bbox_inches='tight')
    
    # Loga no MLflow
    mlflow.log_artifact(tmp_path, artifact_path=os.path.dirname(artifact_name))
    
    # Remove arquivo temporário
    os.remove(tmp_path)