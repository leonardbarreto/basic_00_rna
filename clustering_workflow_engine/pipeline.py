# # import typer
# # from loguru import logger
# # import mlflow

# # from clustering_workflow_engine.modeling.train import train_model

# # app = typer.Typer()

# # @app.command()
# # def run_pipeline(
# #     dataset_name: str = "iris",
# #     model_type: str = "kmeans",  # "kmeans" ou "dbscan"
# #     task: str = "clustering",
# #     n_trials: int = 20
# # ):
# #     """
# #     Pipeline de clustering flexível:
# #     - Suporta KMeans e DBSCAN
# #     - Otimiza hiperparâmetros
# #     - Treina modelo
# #     - Loga métricas e modelo no MLflow
# #     """
# #     logger.info(f"Iniciando pipeline de clustering para {model_type}...")

# #     model, metrics = train_model(
# #         dataset_name=dataset_name,
# #         task="clustering",
# #         model_type=model_type,
# #         n_trials=n_trials
# #     )

# #     # --- Link do run no MLflow ---
# #     client = mlflow.tracking.MlflowClient()
# #     experiment = mlflow.get_experiment_by_name(task.capitalize())
# #     if experiment is not None:
# #         latest_run = client.search_runs(experiment_ids=[experiment.experiment_id],
# #                                         order_by=["start_time DESC"], max_results=1)[0]
# #         run_id = latest_run.info.run_id
# #         tracking_uri = mlflow.get_tracking_uri()
# #         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
# #         logger.success("🎉 Pipeline concluído! Parabéns! 🍺🍺")
# #         logger.success(f"Link para o run no MLflow: {run_link}")
# #     else:
# #         logger.warning("Não foi possível localizar o experimento no MLflow.")

# # if __name__ == "__main__":
# #     app()
# # import typer
# # from loguru import logger
# # import mlflow

# # from clustering_workflow_engine.modeling.train import train_model

# # app = typer.Typer()

# # @app.command()
# # def run_pipeline(
# #     dataset_name: str = "iris",
# #     model_type: str = "kmeans",  # "kmeans" ou "dbscan"
# #     task: str = "clustering",
# #     n_trials: int = 20
# # ):
# #     """
# #     Pipeline de clustering flexível:
# #     - Suporta KMeans e DBSCAN
# #     - Otimiza hiperparâmetros
# #     - Treina modelo
# #     - Loga métricas e modelo no MLflow
# #     """
# #     logger.info(f"Iniciando pipeline de clustering para {model_type}...")

# #     model, metrics = train_model(
# #         dataset_name=dataset_name,
# #         task="clustering",
# #         model_type=model_type,
# #         n_trials=n_trials
# #     )

# #     # --- Link do run no MLflow ---
# #     client = mlflow.tracking.MlflowClient()
# #     experiment = mlflow.get_experiment_by_name(task.capitalize())
# #     if experiment is not None:
# #         latest_run = client.search_runs(experiment_ids=[experiment.experiment_id],
# #                                         order_by=["start_time DESC"], max_results=1)[0]
# #         run_id = latest_run.info.run_id
# #         tracking_uri = mlflow.get_tracking_uri()
# #         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
# #         logger.success("🎉 Pipeline concluído! Parabéns! 🍺🍺")
# #         logger.success(f"Link para o run no MLflow: {run_link}")
# #     else:
# #         logger.warning("Não foi possível localizar o experimento no MLflow.")

# # if __name__ == "__main__":
# #     app()
# # clustering_workflow_engine/pipeline.py
# import typer
# from loguru import logger
# import mlflow
# from mlflow.models.signature import infer_signature
# from pathlib import Path

# from clustering_workflow_engine.config import PROCESSED_DATA_DIR, REPORTS_DIR
# from clustering_workflow_engine.modeling.train import train_model

# app = typer.Typer()

# @app.command()
# def run_pipeline(
#     dataset_name: str = "iris",
#     model_type: str = "kmeans",  # "kmeans" ou "dbscan"
#     task: str = "clustering",
#     n_trials: int = 20
# ):
#     """
#     Pipeline de clustering flexível:
#     - Suporta KMeans e DBSCAN
#     - Otimiza hiperparâmetros
#     - Treina modelo
#     - Loga métricas, dataset, plots e modelo com signature no MLflow
#     """
#     logger.info(f"Iniciando pipeline de clustering para {model_type}...")

#     # --- Treinar modelo ---
#     model, metrics = train_model(
#         dataset_name=dataset_name,
#         task=task,
#         model_type=model_type,
#         n_trials=n_trials
#     )

#     # --- Inferir signature ---
#     processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
#     import pandas as pd
#     df = pd.read_csv(processed_path)
#     X = df.values
#     try:
#         y_pred = model.fit_predict(X) if model_type.lower() == "kmeans" else model.labels_
#         signature = infer_signature(X, y_pred)
#         logger.info("Signature inferida com sucesso.")
#     except Exception as e:
#         signature = None
#         logger.warning(f"Não foi possível inferir signature: {e}")

#     # --- Logar tudo no MLflow ---
#     mlflow.set_experiment(task.capitalize())
#     with mlflow.start_run(run_name=f"{dataset_name}_{model_type}_pipeline") as run:
#         # Log métricas
#         mlflow.log_metrics(metrics)
#         # Log modelo com signature
#         mlflow.sklearn.log_model(model, "model", signature=signature)
#         # Log dataset
#         mlflow.log_artifact(str(processed_path), artifact_path="dataset")
#         # Log plots
#         cluster_plot = REPORTS_DIR / f"{dataset_name}_{model_type}_clusters.png"
#         silhouette_plot = REPORTS_DIR / f"{dataset_name}_{model_type}_silhouette.png"
#         if cluster_plot.exists():
#             mlflow.log_artifact(str(cluster_plot), artifact_path="plots")
#         if silhouette_plot.exists():
#             mlflow.log_artifact(str(silhouette_plot), artifact_path="plots")

#     # --- Link do run no MLflow ---
#     client = mlflow.tracking.MlflowClient()
#     experiment = mlflow.get_experiment_by_name(task.capitalize())
#     if experiment is not None:
#         latest_run = client.search_runs(experiment_ids=[experiment.experiment_id],
#                                         order_by=["start_time DESC"], max_results=1)[0]
#         run_id = latest_run.info.run_id
#         tracking_uri = mlflow.get_tracking_uri()
#         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
#         logger.success("🎉 Pipeline concluído! Parabéns! 🍺🍺")
#         logger.success(f"Link para o run no MLflow: {run_link}")
#     else:
#         logger.warning("Não foi possível localizar o experimento no MLflow.")

# if __name__ == "__main__":
#     app()

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
