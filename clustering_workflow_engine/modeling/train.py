# # # clustering_workflow_engine/modeling/train.py
# # # import os
# # # os.environ["OMP_NUM_THREADS"] = "1"

# # # import pandas as pd
# # # from pathlib import Path
# # # from loguru import logger
# # # import typer
# # # import mlflow
# # # from sklearn.cluster import KMeans, DBSCAN
# # # from sklearn.metrics import silhouette_score

# # # from clustering_workflow_engine.config import PROCESSED_DATA_DIR
# # # from clustering_workflow_engine.modeling.hyperparam_optimization import optimize_clustering
# # # from clustering_workflow_engine.plots import plot_clusters, plot_silhouette_distribution

# # # app = typer.Typer()

# # # def train_model(
# # #     dataset_name: str,
# # #     model_type: str = "kmeans",
# # #     task: str = "clustering",
# # #     n_trials: int = 20
# # # ):
# # #     """
# # #     Treina e otimiza modelo de clustering (KMeans ou DBSCAN), loga métricas e plots no MLflow.
# # #     """
# # #     if task != "clustering":
# # #         raise ValueError("Este train_model está configurado apenas para clustering.")

# # #     # --- Carregar dataset processado ---
# # #     processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
# # #     if not processed_path.exists():
# # #         logger.error(f"Dataset processado não encontrado em {processed_path}")
# # #         raise FileNotFoundError(f"{processed_path} não existe")

# # #     df = pd.read_csv(processed_path)
# # #     X = df.values  # Clustering usa todas as features

# # #     # --- Otimização de hiperparâmetros ---
# # #     logger.info(f"Iniciando otimização de hiperparâmetros para {model_type}...")
# # #     best_params = optimize_clustering(model_type, X, n_trials=n_trials)

# # #     # --- Criar modelo com os melhores parâmetros ---
# # #     if model_type.lower() == "kmeans":
# # #         model = KMeans(**best_params, random_state=42)
# # #     elif model_type.lower() == "dbscan":
# # #         model = DBSCAN(**best_params)
# # #     else:
# # #         raise ValueError(f"Modelo '{model_type}' não suportado")

# # #     # --- Treino ---
# # #     labels = model.fit_predict(X)

# # #     # --- Avaliação ---
# # #     try:
# # #         score = silhouette_score(X, labels)
# # #     except:
# # #         score = None
# # #         logger.warning("Não foi possível calcular silhouette score.")

# # #     metrics = {"silhouette_score": score}

# # #     # --- MLflow tracking ---
# # #     mlflow.set_experiment("Clustering")
# # #     with mlflow.start_run(run_name=model_type):
# # #         # Log params e métricas
# # #         mlflow.log_params(best_params)
# # #         if score is not None:
# # #             mlflow.log_metrics(metrics)
# # #         mlflow.sklearn.log_model(model, "model")

# # #         # --- Gerar e logar plots ---
# # #         fig1 = plot_clusters(X, labels)
# # #         fig2 = plot_silhouette_distribution(X, labels)

# # #         # Salvar temporariamente e logar no MLflow
# # #         fig1_path = Path("clusters_plot.png")
# # #         fig2_path = Path("silhouette_plot.png")
# # #         fig1.savefig(fig1_path)
# # #         fig2.savefig(fig2_path)
# # #         mlflow.log_artifact(str(fig1_path))
# # #         mlflow.log_artifact(str(fig2_path))
# # #         fig1.clf()
# # #         fig2.clf()

# # #     logger.success(f"Treinamento e avaliação concluídos: {metrics}")
# # #     return model, metrics

# # # # --- Typer CLI ---
# # # @app.command()
# # # def main(
# # #     dataset_name: str = "iris",
# # #     model_type: str = "kmeans",
# # #     n_trials: int = 20
# # # ):
# # #     train_model(dataset_name=dataset_name, model_type=model_type, n_trials=n_trials)

# # # if __name__ == "__main__":
# # #     app()

# # import os
# # os.environ["OMP_NUM_THREADS"] = "1"

# # import pandas as pd
# # from pathlib import Path
# # from loguru import logger
# # import typer
# # import mlflow
# # from sklearn.cluster import KMeans, DBSCAN

# # from clustering_workflow_engine.config import PROCESSED_DATA_DIR,REPORTS_DIR
# # from clustering_workflow_engine.modeling.hyperparam_optimization import optimize_clustering
# # from clustering_workflow_engine.modeling.evaluator import evaluate_clustering
# # from clustering_workflow_engine.plots import plot_clusters, plot_silhouette_distribution

# # app = typer.Typer()

# # def train_model(
# #     dataset_name: str,
# #     model_type: str = "kmeans",
# #     task: str = "clustering",
# #     n_trials: int = 20
# # ):
# #     """
# #     Treina e otimiza modelo de clustering (KMeans ou DBSCAN), loga métricas e plots no MLflow.
# #     """
# #     if task != "clustering":
# #         raise ValueError("Este train_model está configurado apenas para clustering.")

# #     # --- Carregar dataset processado ---
# #     processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
# #     if not processed_path.exists():
# #         logger.error(f"Dataset processado não encontrado em {processed_path}")
# #         raise FileNotFoundError(f"{processed_path} não existe")

# #     df = pd.read_csv(processed_path)
# #     X = df.values  # Clustering usa todas as features

# #     # --- Otimização de hiperparâmetros ---
# #     logger.info(f"Iniciando otimização de hiperparâmetros para {model_type}...")
# #     best_params = optimize_clustering(model_type, X, n_trials=n_trials)

# #     # --- Criar modelo com os melhores parâmetros ---
# #     if model_type.lower() == "kmeans":
# #         model = KMeans(**best_params, random_state=42)
# #     elif model_type.lower() == "dbscan":
# #         model = DBSCAN(**best_params)
# #     else:
# #         raise ValueError(f"Modelo '{model_type}' não suportado")

# #     # --- Avaliação ---
# #     metrics, labels = evaluate_clustering(model, X)

# #     # Filtrar métricas None
# #     metrics_valid = {k: v for k, v in metrics.items() if v is not None}
# #     if not metrics_valid:
# #         logger.warning("Nenhuma métrica válida (possível DBSCAN com apenas ruído ou 1 cluster).")
        
# #     # --- MLflow tracking ---
# #     mlflow.set_experiment("Clustering")
# #     with mlflow.start_run(run_name=model_type):
# #         mlflow.log_params(best_params)
# #         if metrics_valid:
# #             mlflow.log_metrics(metrics_valid)
# #         #mlflow.log_metrics(metrics)
# #         mlflow.sklearn.log_model(model, "model")

# #         # --- Gerar e logar plots ---
# #         fig1 = plot_clusters(X, labels)
# #         fig2 = plot_silhouette_distribution(X, labels)

# #         # Garantir que a pasta REPORTS_DIR exista
# #         REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# #         fig1_path = REPORTS_DIR / f"{dataset_name}_clusters.png"
# #         fig2_path = REPORTS_DIR / f"{dataset_name}_silhouette.png"
# #         fig1.savefig(fig1_path)
# #         fig2.savefig(fig2_path)

# #         mlflow.log_artifact(str(fig1_path))
# #         mlflow.log_artifact(str(fig2_path))

# #         fig1.clf()
# #         fig2.clf()

# #     logger.success(f"Treinamento e avaliação concluídos: {metrics}")
# #     return model, metrics

# # # --- Typer CLI ---
# # @app.command()
# # def main(
# #     dataset_name: str = "iris",
# #     model_type: str = "kmeans",
# #     n_trials: int = 20
# # ):
# #     train_model(dataset_name=dataset_name, model_type=model_type, n_trials=n_trials)

# # if __name__ == "__main__":
# #     app()

# import os
# os.environ["OMP_NUM_THREADS"] = "1"  # Evita memory leak no Windows com KMeans/MKL

# import pandas as pd
# from pathlib import Path
# from loguru import logger
# import typer
# import mlflow
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.cluster import KMeans, DBSCAN

# from clustering_workflow_engine.config import PROCESSED_DATA_DIR, REPORTS_DIR
# from clustering_workflow_engine.modeling.hyperparam_optimization import optimize_clustering
# from clustering_workflow_engine.modeling.evaluator import evaluate_clustering
# from clustering_workflow_engine.plots import plot_clusters, plot_silhouette_distribution

# app = typer.Typer()

# def train_model(
#     dataset_name: str,
#     model_type: str = "kmeans",
#     task: str = "clustering",
#     n_trials: int = 20
# ):
#     """
#     Treina e otimiza modelo de clustering (KMeans ou DBSCAN),
#     loga métricas, plots e dataset no MLflow com signature.
#     """
#     if task != "clustering":
#         raise ValueError("Este train_model está configurado apenas para clustering.")

#     # --- Carregar dataset processado ---
#     processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
#     if not processed_path.exists():
#         logger.error(f"Dataset processado não encontrado em {processed_path}")
#         raise FileNotFoundError(f"{processed_path} não existe")

#     df = pd.read_csv(processed_path)
#     X = df.values  # Clustering usa todas as features

#     # --- Otimização de hiperparâmetros ---
#     logger.info(f"Iniciando otimização de hiperparâmetros para {model_type}...")
#     best_params = optimize_clustering(model_type, X, n_trials=n_trials)

#     # --- Criar modelo com os melhores parâmetros ---
#     if model_type.lower() == "kmeans":
#         model = KMeans(**best_params, random_state=42)
#     elif model_type.lower() == "dbscan":
#         model = DBSCAN(**best_params)
#     else:
#         raise ValueError(f"Modelo '{model_type}' não suportado")

#     # --- Avaliação ---
#     metrics, labels = evaluate_clustering(model, X)

#     # Filtrar métricas None
#     metrics_valid = {k: v for k, v in metrics.items() if v is not None}
#     if not metrics_valid:
#         logger.warning("Nenhuma métrica válida (possível DBSCAN com apenas ruído ou 1 cluster).")

#     # --- MLflow tracking ---
#     mlflow.set_experiment("Clustering")
#     with mlflow.start_run(run_name=model_type):
#         # Log params e métricas
#         mlflow.log_params(best_params)
#         if metrics_valid:
#             mlflow.log_metrics(metrics_valid)

#         # --- Log dataset com signature ---
#         signature = infer_signature(df)
#         mlflow.sklearn.log_model(model, "model", signature=signature)
#         mlflow.log_artifact(processed_path, artifact_path="data")

#         # --- Gerar e logar plots ---
#         fig1 = plot_clusters(X, labels)
#         fig2 = plot_silhouette_distribution(X, labels)

#         # Garantir que a pasta REPORTS_DIR exista
#         REPORTS_DIR.mkdir(parents=True, exist_ok=True)

#         fig1_path = REPORTS_DIR / f"{dataset_name}_clusters.png"
#         fig2_path = REPORTS_DIR / f"{dataset_name}_silhouette.png"
#         fig1.savefig(fig1_path)
#         fig2.savefig(fig2_path)

#         mlflow.log_artifact(str(fig1_path))
#         mlflow.log_artifact(str(fig2_path))

#         fig1.clf()
#         fig2.clf()

#     logger.success(f"Treinamento e avaliação concluídos: {metrics}")
#     return model, metrics

# # --- Typer CLI ---
# @app.command()
# def main(
#     dataset_name: str = "iris",
#     model_type: str = "kmeans",
#     n_trials: int = 20
# ):
#     train_model(dataset_name=dataset_name, model_type=model_type, n_trials=n_trials)

# if __name__ == "__main__":
#     app()

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Evita memory leak no Windows com KMeans/MKL

from pathlib import Path
import typer
import pandas as pd
from loguru import logger
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans, DBSCAN
from mlflow.models.signature import infer_signature

from clustering_workflow_engine.config import PROCESSED_DATA_DIR, REPORTS_DIR
from clustering_workflow_engine.modeling.hyperparam_optimization import optimize_clustering
from clustering_workflow_engine.modeling.evaluator import evaluate_clustering
from clustering_workflow_engine.plots import plot_clusters, plot_silhouette_distribution

app = typer.Typer()

def train_model(
    dataset_name: str,
    model_type: str = "kmeans",
    task: str = "clustering",
    n_trials: int = 20
):
    """
    Treina e otimiza modelo de clustering (KMeans ou DBSCAN), loga métricas, plots e dataset no MLflow.
    """
    if task != "clustering":
        raise ValueError("Este train_model está configurado apenas para clustering.")

    # --- Carregar dataset processado ---
    processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    if not processed_path.exists():
        logger.error(f"Dataset processado não encontrado em {processed_path}")
        raise FileNotFoundError(f"{processed_path} não existe")

    df = pd.read_csv(processed_path)
    X = df.values  # Clustering usa todas as features

    # --- Otimização de hiperparâmetros ---
    logger.info(f"Iniciando otimização de hiperparâmetros para {model_type}...")
    best_params = optimize_clustering(model_type, X, n_trials=n_trials)

    # --- Criar modelo ---
    if model_type.lower() == "kmeans":
        model = KMeans(**best_params, random_state=42)
    elif model_type.lower() == "dbscan":
        model = DBSCAN(**best_params)
    else:
        raise ValueError(f"Modelo '{model_type}' não suportado")

    # --- Avaliação ---
    metrics, labels = evaluate_clustering(model, X)

    metrics_valid = {k: v for k, v in metrics.items() if v is not None}
    if not metrics_valid:
        logger.warning("Nenhuma métrica válida (possível DBSCAN com apenas ruído ou 1 cluster).")

    # --- MLflow tracking ---
    mlflow.set_experiment("Clustering")
    with mlflow.start_run(run_name=f"{model_type}_{dataset_name}"):

        # Log params e métricas
        mlflow.log_params(best_params)
        if metrics_valid:
            mlflow.log_metrics(metrics_valid)

        # Converter colunas inteiras para float
        df_float = df.astype(float)
        # Log modelo com signature
        signature = infer_signature(df_float)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        # Log modelo com signature
        # signature = infer_signature(df)
        # mlflow.sklearn.log_model(model, "model", signature=signature)

        # --- Gerar e logar plots ---
        fig1 = plot_clusters(X, labels)
        fig2 = plot_silhouette_distribution(X, labels)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig1_path = REPORTS_DIR / f"{dataset_name}_clusters.png"
        fig2_path = REPORTS_DIR / f"{dataset_name}_silhouette.png"
        fig1.savefig(fig1_path)
        fig2.savefig(fig2_path)
        mlflow.log_artifact(str(fig1_path))
        mlflow.log_artifact(str(fig2_path))
        fig1.clf()
        fig2.clf()

        # --- Log dataset original ---
        dataset_path = REPORTS_DIR / f"{dataset_name}_dataset.csv"
        df.to_csv(dataset_path, index=False)
        mlflow.log_artifact(str(dataset_path))

    logger.success(f"Treinamento e avaliação concluídos: {metrics}")
    return model, metrics

# --- Typer CLI ---
@app.command()
def main(
    dataset_name: str = "wine",
    model_type: str = "kmeans",
    n_trials: int = 20
):
    train_model(dataset_name=dataset_name, model_type=model_type, n_trials=n_trials)

if __name__ == "__main__":
    app()