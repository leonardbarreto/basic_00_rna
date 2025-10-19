# # ai_workflow_engine/train.py
# from pathlib import Path
# import typer
# from loguru import logger
# import mlflow
# from sklearn.model_selection import train_test_split
# import pandas as pd

# from ai_workflow_engine.experiments.mlflow_utils import start_experiment, log_params, log_metrics, log_model
# from ai_workflow_engine.modeling.hyperparam_optimization import optimize_model
# from ai_workflow_engine.modeling.evaluator import evaluate_model
# from ai_workflow_engine.config import PROCESSED_DATA_DIR
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# app = typer.Typer()

# def train_model(
#     dataset_name: str,
#     model_type: str = "random_forest",
#     task: str = "classification",
#     test_size: float = 0.2,
#     cv: int = 3,
#     n_trials: int = 50
# ):
#     """
#     Função que realiza treino, otimização de hiperparâmetros e avaliação.
#     Pode ser chamada de pipeline ou via terminal.
#     """
#     # --- Carregar dataset processado ---
#     processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
#     if not processed_path.exists():
#         logger.error(f"Dataset processado não encontrado em {processed_path}")
#         raise FileNotFoundError(f"{processed_path} não existe")

#     df = pd.read_csv(processed_path)
#     if "target" not in df.columns:
#         raise ValueError("O dataset deve conter a coluna 'target'")

#     X = df.drop(columns=["target"])
#     y = df["target"]

#     # --- Seleção da classe de modelo ---
#     if task == "classification":
#         model_class = RandomForestClassifier
#     elif task == "regression":
#         model_class = RandomForestRegressor
#     else:
#         raise ValueError(f"Tarefa '{task}' não suportada")

#     # --- Otimização de hiperparâmetros ---
#     logger.info("Iniciando otimização de hiperparâmetros...")
#     best_params = optimize_model(model_class, X, y, task, n_trials=n_trials, cv=cv)
#     logger.info(f"Melhores parâmetros encontrados: {best_params}")

#     # --- Criar modelo com melhores parâmetros ---
#     model = model_class(**best_params)

#     # --- MLflow Tracking ---
#     start_experiment(task.capitalize())
#     with mlflow.start_run(run_name=model_type.capitalize()):
#         log_params(best_params)

#         # --- Divisão treino/teste ---
#         stratify = y if task == "classification" else None
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, stratify=stratify
#         )

#         # --- Treino ---
#         model.fit(X_train, y_train)

#         # --- Avaliação ---
#         metrics = evaluate_model(model, X_test, y_test, task)
#         log_metrics(metrics)
#         log_model(model)

#     logger.success(f"Treinamento e avaliação concluídos: {metrics} 🍻🍻")
#     return model, metrics

# # --- Typer CLI ---
# @app.command()
# def main(
#     dataset_name: str = "iris",
#     model_type: str = "random_forest",
#     task: str = "classification",
#     test_size: float = 0.2,
#     cv: int = 3,
#     n_trials: int = 50
# ):
#     train_model(
#         dataset_name=dataset_name,
#         model_type=model_type,
#         task=task,
#         test_size=test_size,
#         cv=cv,
#         n_trials=n_trials
#     )

# if __name__ == "__main__":
#     app()

# ai_workflow_engine/train.py
from pathlib import Path
import typer
from loguru import logger
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ai_workflow_engine.experiments.mlflow_utils import start_experiment, log_params, log_metrics, log_model
from ai_workflow_engine.modeling.hyperparam_optimization import optimize_model
from ai_workflow_engine.modeling.evaluator import evaluate_model
from ai_workflow_engine.config import PROCESSED_DATA_DIR

import optuna
import logging

app = typer.Typer()

def train_model(
    dataset_name: str,
    model_type: str = "random_forest",
    task: str = "classification",
    test_size: float = 0.2,
    cv: int = 3,
    n_trials: int = 50
):
    """
    Função que realiza treino, otimização de hiperparâmetros e avaliação.
    Pode ser chamada de pipeline ou via terminal.
    """
    # --- Carregar dataset processado ---
    processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    if not processed_path.exists():
        logger.error(f"Dataset processado não encontrado em {processed_path}")
        raise FileNotFoundError(f"{processed_path} não existe")

    df = pd.read_csv(processed_path)
    if "target" not in df.columns:
        raise ValueError("O dataset deve conter a coluna 'target'")

    X = df.drop(columns=["target"])
    y = df["target"]

    # --- Seleção da classe de modelo ---
    if task == "classification":
        model_class = RandomForestClassifier
    elif task == "regression":
        model_class = RandomForestRegressor
    else:
        raise ValueError(f"Tarefa '{task}' não suportada")

    # --- Mensagem inicial e suprimir logs do Optuna ---
    logger.info("Iniciando otimização de hiperparâmetros...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    # --- Otimização ---
    best_params = optimize_model(model_class, X, y, task, n_trials=n_trials, cv=cv)
    logger.info(f"Melhores parâmetros encontrados: {best_params}")

    # --- Criar modelo com melhores parâmetros ---
    model = model_class(**best_params)

    # --- MLflow Tracking ---
    start_experiment(task.capitalize())
    with mlflow.start_run(run_name=model_type.capitalize()):
        log_params(best_params)

        # --- Divisão treino/teste ---
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify
        )

        # --- Treino ---
        model.fit(X_train, y_train)

        # --- Avaliação ---
        metrics = evaluate_model(model, X_test, y_test, task)
        log_metrics(metrics)
        log_model(model)

    logger.success(f"Treinamento e avaliação concluídos: {metrics}")
    return model, metrics

# --- Typer CLI ---
@app.command()
def main(
    dataset_name: str = "iris",
    model_type: str = "random_forest",
    task: str = "classification",
    test_size: float = 0.2,
    cv: int = 3,
    n_trials: int = 50
):
    train_model(
        dataset_name=dataset_name,
        model_type=model_type,
        task=task,
        test_size=test_size,
        cv=cv,
        n_trials=n_trials
    )

if __name__ == "__main__":
    app()
