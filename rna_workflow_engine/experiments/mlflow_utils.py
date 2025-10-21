import io
import os
import tempfile

import mlflow
from loguru import logger
from matplotlib.figure import Figure

# def start_experiment(experiment_name: str):
#     """Inicia ou retorna um experimento MLflow"""
#     mlflow.set_experiment(experiment_name)
#     logger.info(f"Experimento MLflow iniciado: {experiment_name}")


def start_experiment(experiment_name: str, run_name: str = None):
    """
    Inicia (ou recupera) um experimento no MLflow e abre um run ativo.
    """
    mlflow.set_experiment(experiment_name)
    # Fecha run ativo se existir
    if mlflow.active_run() is not None:
        mlflow.end_run()

    active_run = mlflow.start_run(run_name=run_name)
    logger.info(
        f"Experimento MLflow iniciado: {experiment_name} | Run: {run_name or active_run.info.run_id}")
    return active_run


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


def get_active_run_id():
    """
    Retorna o run_id do run MLflow ativo.
    """
    run = mlflow.active_run()
    if run is not None:
        return run.info.run_id
    return None


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
