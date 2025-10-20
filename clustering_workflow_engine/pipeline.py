import typer
from loguru import logger
import mlflow

from clustering_workflow_engine.modeling.train import train_model
from clustering_workflow_engine.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def run_pipeline(
    dataset_name: str = "wine",
    model_type: str = "kmeans",  # "kmeans" ou "dbscan"
    task: str = "clustering",
    n_trials: int = 20
):
    """
    Pipeline de clustering para o dataset Wine:
    - Otimiza hiperparâmetros
    - Treina modelo
    - Loga métricas, modelo, dataset e plots no MLflow
    - Exibe link do run
    """
    logger.info(f"Iniciando pipeline de clustering para {model_type} no dataset '{dataset_name}'...")

    # Treinamento e log no MLflow
    model, metrics = train_model(
        dataset_name=dataset_name,
        model_type=model_type,
        task=task,
        n_trials=n_trials
    )

    # --- Link do run no MLflow ---
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("Clustering")
    if experiment is not None:
        latest_run = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )[0]
        run_id = latest_run.info.run_id
        tracking_uri = mlflow.get_tracking_uri()
        run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
        logger.success("🎉 Pipeline concluído com sucesso! 🍺")
        logger.success(f"Link para o run no MLflow: {run_link}")
    else:
        logger.warning("Não foi possível localizar o experimento no MLflow.")

if __name__ == "__main__":
    app()
