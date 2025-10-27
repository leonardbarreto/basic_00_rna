# pipeline.py
from pathlib import Path

import torch
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from rna_workflow_engine.config import FIGURES_DIR
from rna_workflow_engine.dataset import fetch_dataset
from rna_workflow_engine.experiments.mlflow_utils import (end_run, log_figure,
                                                          log_metrics,
                                                          log_model,
                                                          log_params,
                                                          start_run)
from rna_workflow_engine.modeling.evaluator import evaluate_model
from rna_workflow_engine.modeling.hyperparam_optimization import \
    optimize_hyperparams
from rna_workflow_engine.modeling.train import train_model
from rna_workflow_engine.plots import (get_classification_preds,
                                       plot_confusion_matrix, plot_loss_curve)

app = typer.Typer()


@app.command()
def run_pipeline(
    dataset_name: str = typer.Option(
        "iris", help="Nome do dataset: iris, diabetes, titanic, wine."
    ),
    # dataset_name: str = typer.Option(..., help="Nome do dataset"),
    experiment_name: str = typer.Option(
        "MLP", help="Nome do experimento MLflow"),
    run_name: str = typer.Option(None, help="Nome do run MLflow"),
    optimize_hyperparams_flag: bool = typer.Option(
        True, help="Executar otimização de hiperparâmetros")
):
    logger.info(f"Iniciando pipeline para dataset '{dataset_name}'")

    # 1️⃣ Carrega dataset
    X, y = fetch_dataset(dataset_name)
    input_dim = X.shape[1]
    output_dim = len(y.unique()) if len(y.unique()) > 2 else 1

    # Split treino/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2️⃣ Inicia run MLflow
    run = start_run(experiment_name=experiment_name, run_name=run_name)

    try:
        # 3️⃣ Otimização de hiperparâmetros
        if optimize_hyperparams_flag:
            logger.info("Otimizando hiperparâmetros...")
            best_params = optimize_hyperparams(X_train, y_train, n_trials=20)
        else:
            # Valores default
            best_params = {"hidden1": 64, "hidden2": 32,
                           "dropout": 0.2, "lr": 1e-3}

        log_params(best_params)

        # 4️⃣ Treinamento final
        logger.info("Treinando modelo final...")
        model, train_losses, val_losses = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden1=best_params["hidden1"],
            hidden2=best_params["hidden2"],
            dropout=best_params["dropout"],
            lr=best_params["lr"],
            epochs=50,
            batch_size=32,
            device="cpu"
        )

        # 5️⃣ Logging de métricas usando evaluator
        metrics, y_val_preds = evaluate_model(
            model=model,
            X=X_val,
            y=y_val,
            task="classification",
            device="cpu"
        )
        log_metrics(metrics)
        logger.info(f"Métricas de validação logadas: {metrics}")

        # 6️⃣ Logging do modelo
        log_model(
            model=model,
            model_name=f"{dataset_name}_MLP_Model",
            X_sample=X_val.head(5),
            y_sample=y_val_preds[:5],
            metadata={
                "dataset": dataset_name,
                "task": "classification",
                "metrics": metrics
            }
        )

        # 7️⃣ Geração de plots
        loss_plot_path = FIGURES_DIR / f"{dataset_name}_loss_curve.png"
        plot_loss_curve({"train_loss": train_losses,
                        "val_loss": val_losses}, save_path=loss_plot_path)

        cm_plot_path = FIGURES_DIR / f"{dataset_name}_confusion_matrix.png"
        plot_confusion_matrix(y_val, y_val_preds, save_path=cm_plot_path)

        log_figure(path=loss_plot_path)
        log_figure(path=cm_plot_path)

    finally:
        # 8️⃣ Finaliza run MLflow
        end_run()
        logger.success("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    app()
