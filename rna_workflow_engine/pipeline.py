# # # --- pipeline.py ---
# # import typer
# # from loguru import logger
# # import mlflow

# # from rna_workflow_engine.modeling.train import train

# # app = typer.Typer()


# # @app.command()
# # def run_pipeline(
# #     dataset_name: str = "iris",
# #     n_trials: int = 20
# # ):
# #     """
# #     Pipeline para treinamento de MLP:
# #     - Otimiza hiperparâmetros via Optuna
# #     - Treina modelo com cross-validation
# #     - Loga métricas, modelo e parâmetros no MLflow
# #     - Exibe link do run
# #     """
# #     logger.info(f"Iniciando pipeline para dataset '{dataset_name}' com {n_trials} trials...")

# #     # --- Treinamento e otimização ---
# #     model, best_params, metrics_list = train(dataset_name=dataset_name, n_trials=n_trials)

# #     # --- Link do run no MLflow ---
# #     client = mlflow.tracking.MlflowClient()
# #     experiment = mlflow.get_experiment_by_name("MLP_Regression")
# #     if experiment is not None:
# #         latest_run = client.search_runs(
# #             experiment_ids=[experiment.experiment_id],
# #             order_by=["start_time DESC"],
# #             max_results=1
# #         )[0]
# #         run_id = latest_run.info.run_id
# #         tracking_uri = mlflow.get_tracking_uri()
# #         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
# #         logger.success("🎉 Pipeline concluído com sucesso! 🍺")
# #         logger.success(f"Link para o run no MLflow: {run_link}")
# #     else:
# #         logger.warning("Não foi possível localizar o experimento no MLflow.")


# # if __name__ == "__main__":
# #     app()

# # # --- pipeline.py ---
# # import typer
# # from loguru import logger
# # import mlflow

# # from rna_workflow_engine.modeling.train import train
# # from rna_workflow_engine.modeling.evaluator import evaluate_model

# # app = typer.Typer()


# # @app.command()
# # def run_pipeline(
# #     dataset_name: str = "iris",
# #     n_trials: int = 20,
# #     test_split: float = 0.2
# # ):
# #     """
# #     Executa pipeline completo:
# #     - Treina e valida modelo MLP (Optuna + Cross-Validation)
# #     - Avalia no conjunto de teste
# #     - Loga tudo no mesmo run do MLflow
# #     """
# #     experiment_name = "MLP_Regression"
# #     mlflow.set_experiment(experiment_name)

# #     with mlflow.start_run(run_name=f"{dataset_name}_MLP") as run:
# #         logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

# #         # --- Treinamento + Validação ---
# #         model_path, best_params, metrics_list,_ = train(
# #             dataset_name=dataset_name,
# #             n_trials=n_trials,
# #             test_split=test_split
# #         )

# #         # Logar médias das métricas de validação
# #         mean_val_loss = sum(m["val_loss"] for m in metrics_list) / len(metrics_list)
# #         mlflow.log_metric("mean_val_loss", mean_val_loss)

# #         # Logar hiperparâmetros ótimos
# #         mlflow.log_params(best_params)

# #         # --- Avaliação Final (Teste) ---
# #         test_metrics = evaluate_model(model_path=model_path, dataset_name=dataset_name)
# #         mlflow.log_metrics(test_metrics)

# #         logger.success("✅ Pipeline completo: treino, validação e teste no mesmo run!")
# #         logger.info(f"MLflow Run ID: {run.info.run_id}")
# #         logger.info(f"MLflow Experiment: {experiment_name}")


# # if __name__ == "__main__":
# #     app()

# # import mlflow
# # # --- pipeline.py ---
# # import typer
# # from loguru import logger

# # from rna_workflow_engine.modeling.evaluator import evaluate_model
# # from rna_workflow_engine.modeling.train import train

# # app = typer.Typer()


# # @app.command()
# # def run_pipeline(
# #     dataset_name: str = "iris",
# #     n_trials: int = 20,
# #     test_split: float = 0.2
# # ):
# #     """
# #     Pipeline completo para MLP:
# #     - Otimiza hiperparâmetros via Optuna
# #     - Treina com cross-validation
# #     - Avalia no conjunto de teste
# #     - Loga métricas, modelo e parâmetros no MLflow
# #     - Exibe link do run
# #     """
# #     logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

# #     # --- Treinamento + Validação ---
# #     model_path, best_params, metrics_list = train(
# #         dataset_name=dataset_name,
# #         n_trials=n_trials,
# #         test_split=test_split
# #     )

# #     # --- Logar médias das métricas de validação ---
# #     mean_val_loss = sum(m["val_loss"]
# #                         for m in metrics_list) / len(metrics_list)
# #     mlflow.log_metric("mean_val_loss", mean_val_loss)
# #     logger.info(f"Val Loss médio (CV): {mean_val_loss:.4f}")

# #     # --- Avaliação Final (Teste) ---
# #     test_metrics = evaluate_model(
# #         model_path=model_path,
# #         dataset_name=dataset_name,
# #         h1=best_params['h1'],
# #         h2=best_params['h2'],
# #         dropout=best_params['dropout']
# #     )
# #     mlflow.log_metrics(test_metrics)
# #     logger.info(f"Test Metrics: {test_metrics}")

# #     # --- Link do run no MLflow ---
# #     client = mlflow.tracking.MlflowClient()
# #     experiment = mlflow.get_experiment_by_name("MLP_Regression")
# #     if experiment is not None:
# #         latest_run = client.search_runs(
# #             experiment_ids=[experiment.experiment_id],
# #             order_by=["start_time DESC"],
# #             max_results=1
# #         )[0]
# #         run_id = latest_run.info.run_id
# #         tracking_uri = mlflow.get_tracking_uri()
# #         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
# #         logger.success("🎉 Pipeline concluído com sucesso! 🍺")
# #         logger.success(f"Link para o run no MLflow: {run_link}")
# #     else:
# #         logger.warning("Não foi possível localizar o experimento no MLflow.")


# # if __name__ == "__main__":
# #     app()

# # --- pipeline.py ---
# import typer
# from loguru import logger
# import mlflow
# import matplotlib.pyplot as plt
# from rna_workflow_engine.modeling.evaluator import evaluate_model_from_path
# from rna_workflow_engine.modeling.train import train

# app = typer.Typer()


# @app.command()
# def run_pipeline(
#     dataset_name: str = "iris",
#     n_trials: int = 20,
#     test_split: float = 0.2
# ):
#     """
#     Pipeline completo para MLP:
#     - Otimiza hiperparâmetros via Optuna
#     - Treina com cross-validation
#     - Avalia no conjunto de teste
#     - Loga métricas, modelo e parâmetros no MLflow
#     - Exibe link do run
#     """
#     logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

#     # --- Treinamento + Validação ---
#     model_path, best_params, metrics_list, train_losses_history, val_losses_history = train(
#         dataset_name=dataset_name,
#         n_trials=n_trials,
#         test_split=test_split
#     )

#     # --- Logar médias das métricas de validação ---
#     mean_val_loss = sum(m["val_loss"] for m in metrics_list) / len(metrics_list)
#     mlflow.log_metric("mean_val_loss", mean_val_loss)
#     logger.info(f"Val Loss médio (CV): {mean_val_loss:.4f}")

#     # --- Avaliação Final (Teste) ---
#     test_metrics = evaluate_model_from_path(
#         model_class=best_params["model_class"],  # a classe do modelo deve vir do train
#         model_path=model_path,
#         X_test=best_params["X_test"],
#         y_test=best_params["y_test"],
#         device=best_params.get("device", "cpu"),
#         **best_params
#     )
#     mlflow.log_metrics(test_metrics)
#     logger.info(f"Test Metrics: {test_metrics}")

#     # --- Gráfico Treino x Validação vs Epochs ---
#     plt.figure(figsize=(10, 6))
#     for i, (train_loss, val_loss) in enumerate(zip(train_losses_history, val_losses_history)):
#         plt.plot(train_loss, label=f"Train Fold {i+1}", linestyle="--")
#         plt.plot(val_loss, label=f"Val Fold {i+1}")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss (MSE)")
#     plt.title("Treino x Validação vs Epochs")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("train_val_losses.png")
#     mlflow.log_artifact("train_val_losses.png")
#     plt.show()

#     # --- Link do run no MLflow ---
#     client = mlflow.tracking.MlflowClient()
#     experiment = mlflow.get_experiment_by_name("MLP_Regression")
#     if experiment is not None:
#         latest_run = client.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             order_by=["start_time DESC"],
#             max_results=1
#         )[0]
#         run_id = latest_run.info.run_id
#         tracking_uri = mlflow.get_tracking_uri()
#         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
#         logger.success("🎉 Pipeline concluído com sucesso! 🍺")
#         logger.success(f"Link para o run no MLflow: {run_link}")
#     else:
#         logger.warning("Não foi possível localizar o experimento no MLflow.")


# if __name__ == "__main__":
#     app()

# # --- pipeline.py ---
# import mlflow
# import typer
# from loguru import logger
# import matplotlib.pyplot as plt

# from rna_workflow_engine.modeling.evaluator import evaluate_model
# from rna_workflow_engine.modeling.train import train

# app = typer.Typer()

# def plot_training_history(train_losses, val_losses, dataset_name="dataset"):
#     """Gera gráfico de treino vs validação e loga no MLflow"""
#     plt.figure(figsize=(8, 5))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Train vs Validation Loss - {dataset_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Salvar figura temporariamente
#     fig_path = f"train_val_loss_{dataset_name}.png"
#     plt.savefig(fig_path)
#     plt.close()
    
#     # Logar no MLflow
#     mlflow.log_artifact(fig_path)
#     logger.info(f"📊 Gráfico de treino vs validação salvo e logado: {fig_path}")


# @app.command()
# def run_pipeline(
#     dataset_name: str = "iris",
#     n_trials: int = 20,
#     test_split: float = 0.2
# ):
#     """
#     Pipeline completo para MLP:
#     - Otimiza hiperparâmetros via Optuna
#     - Treina com cross-validation
#     - Avalia no conjunto de teste
#     - Loga métricas, modelo e parâmetros no MLflow
#     - Exibe link do run
#     """
#     logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

#     # --- Treinamento + Validação ---
#     model_path, best_params, metrics_list, train_losses_history, val_losses_history = train(
#         dataset_name=dataset_name,
#         n_trials=n_trials,
#         test_split=test_split
#     )

#     # --- Logar médias das métricas de validação ---
#     mean_val_loss = sum(m["val_loss"] for m in metrics_list) / len(metrics_list)
#     mlflow.log_metric("mean_val_loss", mean_val_loss)
#     logger.info(f"Val Loss médio (CV): {mean_val_loss:.4f}")

#     # --- Gráfico de Treino vs Validação ---
#     plot_training_history(train_losses_history, val_losses_history, dataset_name)

#     # --- Avaliação Final (Teste) ---
#     test_metrics = evaluate_model(
#         model_path=model_path,
#         dataset_name=dataset_name,
#         h1=best_params['h1'],
#         h2=best_params['h2'],
#         dropout=best_params['dropout']
#     )
#     mlflow.log_metrics(test_metrics)
#     logger.info(f"Test Metrics: {test_metrics}")

#     # --- Link do run no MLflow ---
#     client = mlflow.tracking.MlflowClient()
#     experiment = mlflow.get_experiment_by_name("MLP_Regression")
#     if experiment is not None:
#         latest_run = client.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             order_by=["start_time DESC"],
#             max_results=1
#         )[0]
#         run_id = latest_run.info.run_id
#         tracking_uri = mlflow.get_tracking_uri()
#         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
#         logger.success("🎉 Pipeline concluído com sucesso! 🍺")
#         logger.success(f"Link para o run no MLflow: {run_link}")
#     else:
#         logger.warning("Não foi possível localizar o experimento no MLflow.")


# if __name__ == "__main__":
#     app()

# # --- pipeline.py ---
# import os
# import typer
# from loguru import logger
# import mlflow

# from rna_workflow_engine.modeling.train import train

# app = typer.Typer()

# @app.command()
# def run_pipeline(
#     dataset_name: str = "iris",
#     n_trials: int = 20,
#     test_split: float = 0.2
# ):
#     """
#     Pipeline completo para MLP em PyTorch:
#     - Otimiza hiperparâmetros via Optuna
#     - Treina com cross-validation
#     - Avalia no conjunto de teste
#     - Loga métricas, figuras e modelo no MLflow
#     - Exibe link do run
#     """
#     logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

#     # --- Treinamento + CV + Teste ---
#     model_path, best_params, metrics_list = train(
#         dataset_name=dataset_name,
#         n_trials=n_trials,
#         test_split=test_split
#     )

#     # --- Logar métricas médias de validação ---
#     mean_val_loss = sum(m["val_loss"] for m in metrics_list) / len(metrics_list)
#     mlflow.log_metric("mean_val_loss", mean_val_loss)
#     logger.info(f"Val Loss médio (CV): {mean_val_loss:.4f}")

#     # --- Link do run no MLflow ---
#     client = mlflow.tracking.MlflowClient()
#     experiment = mlflow.get_experiment_by_name("MLP_Regression")
#     if experiment is not None:
#         latest_run = client.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             order_by=["start_time DESC"],
#             max_results=1
#         )[0]
#         run_id = latest_run.info.run_id
#         tracking_uri = mlflow.get_tracking_uri()
#         run_link = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
#         logger.success("🎉 Pipeline concluído com sucesso! 🍺")
#         logger.success(f"Link para o run no MLflow: {run_link}")
#     else:
#         logger.warning("Não foi possível localizar o experimento no MLflow.")

# if __name__ == "__main__":
#     app()

import os
import typer
from loguru import logger
import mlflow

from rna_workflow_engine.modeling.train import train

app = typer.Typer()
EXPERIMENT_NAME = "MLP_Regression"

@app.command()
def run_pipeline(
    dataset_name: str = "iris",
    n_trials: int = 20,
    test_split: float = 0.2
):
    """
    Pipeline completo para MLP em PyTorch:
    - Otimiza hiperparâmetros via Optuna
    - Treina com cross-validation
    - Avalia no conjunto de teste
    - Loga métricas, figuras e modelo no MLflow
    - Exibe link do run
    """
    logger.info(f"🚀 Iniciando pipeline para dataset '{dataset_name}'")

    # --- Criar ou recuperar experimento ---
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info(f"Experimento '{EXPERIMENT_NAME}' criado com ID {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Usando experimento existente '{EXPERIMENT_NAME}' com ID {experiment_id}")

    # --- Iniciar run ---
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # --- Treinamento + CV + Teste ---
        model_path, best_params, metrics_list = train(
            dataset_name=dataset_name,
            n_trials=n_trials,
            test_split=test_split
        )

        # --- Logar métricas médias de validação ---
        mean_val_loss = sum(m["val_loss"] for m in metrics_list) / len(metrics_list)
        mlflow.log_metric("mean_val_loss", mean_val_loss)
        logger.info(f"Val Loss médio (CV): {mean_val_loss:.4f}")

        # --- Logar parâmetros ---
        mlflow.log_params(best_params)

        # --- Link do run ---
        tracking_uri = mlflow.get_tracking_uri()
        run_link = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        logger.success("🎉 Pipeline concluído com sucesso! 🍺")
        logger.success(f"Link para o run no MLflow: {run_link}")

if __name__ == "__main__":
    app()
