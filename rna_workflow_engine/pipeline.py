# --- pipeline.py ---
import typer
from loguru import logger
import mlflow

from rna_workflow_engine.modeling.train import train

app = typer.Typer()


@app.command()
def run_pipeline(
    dataset_name: str = "iris",
    n_trials: int = 20
):
    """
    Pipeline para treinamento de MLP:
    - Otimiza hiperparâmetros via Optuna
    - Treina modelo com cross-validation
    - Loga métricas, modelo e parâmetros no MLflow
    - Exibe link do run
    """
    logger.info(f"Iniciando pipeline para dataset '{dataset_name}' com {n_trials} trials...")

    # --- Treinamento e otimização ---
    model, best_params, metrics_list = train(dataset_name=dataset_name, n_trials=n_trials)

    # --- Link do run no MLflow ---
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("MLP_Regression")
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
