import optuna
import torch
# rna_workflow_engine/modeling/hyperparam_optimization.py
import typer
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rna_workflow_engine.modeling.train import train_model

app = typer.Typer()


def objective(trial, X, y):
    """
    Função objetivo do Optuna — ajusta os hiperparâmetros do MLP
    e retorna a acurácia de validação.
    """
    params = {
        "hidden1": trial.suggest_int("hidden1", 32, 128),
        "hidden2": trial.suggest_int("hidden2", 16, 64),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }

    # Split dos dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    input_dim = X_train.shape[1]
    output_dim = len(y.unique()) if len(y.unique()) > 2 else 1

    # Treinamento curto para avaliação de performance
    model, train_losses, val_losses = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        output_dim=output_dim,
        task="classification",
        hidden1=params["hidden1"],
        hidden2=params["hidden2"],
        dropout=params["dropout"],
        lr=params["lr"],
        epochs=10,
        batch_size=32,
        device="cpu"
    )

    # Predição e cálculo da acurácia de validação
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        logits = model(X_val_tensor)
        if output_dim == 1:
            preds = (torch.sigmoid(logits).squeeze() >= 0.5).numpy()
        else:
            preds = torch.argmax(logits, axis=1).numpy()

    val_acc = accuracy_score(y_val, preds)
    logger.info(
        f"Trial concluído — Val Accuracy: {val_acc:.4f} | Params: {params}")

    return val_acc


def optimize_hyperparams(X, y, n_trials: int = 20):
    """
    Executa a otimização de hiperparâmetros usando Optuna.
    """
    logger.info("Iniciando otimização de hiperparâmetros com Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    logger.success(f"Melhores parâmetros encontrados: {study.best_params}")
    return study.best_params


@app.command()
def run_optimization(
    dataset_name: str = typer.Option(
        ..., help="Nome do dataset (ex: iris, titanic, wine, diabetes)"),
    n_trials: int = typer.Option(
        20, help="Número de tentativas (trials) de otimização")
):
    """
    CLI Typer para executar a otimização via terminal.
    """
    from rna_workflow_engine.data.dataset import fetch_dataset
    X, y = fetch_dataset(dataset_name)
    best_params = optimize_hyperparams(X, y, n_trials=n_trials)
    typer.echo(
        f"Melhores parâmetros encontrados para {dataset_name}: {best_params}")


if __name__ == "__main__":
    app()
