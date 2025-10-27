# # # import torch
# # # from torch.utils.data import TensorDataset, DataLoader, random_split
# # # import optuna
# # # from loguru import logger
# # # from rna_workflow_engine.utils import get_device

# # # # --- Exemplo de MLP ---
# # # import torch.nn as nn
# # # class MLP(nn.Module):
# # #     def __init__(self, input_dim, output_dim, h1=32, h2=32, dropout=0.2):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(input_dim, h1),
# # #             nn.ReLU(),
# # #             nn.Dropout(dropout),
# # #             nn.Linear(h1, h2),
# # #             nn.ReLU(),
# # #             nn.Dropout(dropout),
# # #             nn.Linear(h2, output_dim)
# # #         )
# # #     def forward(self, x):
# # #         return self.net(x)

# # # # --- Função objetivo para Optuna ---
# # # def objective(trial, X, y, n_epochs=50, val_split=0.2):
# # #     # Espaço de busca
# # #     h1 = trial.suggest_int("h1", 16, 128)
# # #     h2 = trial.suggest_int("h2", 16, 128)
# # #     dropout = trial.suggest_float("dropout", 0.0, 0.5)
# # #     optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
# # #     lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
# # #     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

# # #     device = get_device()
# # #     input_dim = X.shape[1]
# # #     output_dim = 1  # ajuste para classificação se necessário

# # #     # Criar modelo
# # #     model = MLP(input_dim=input_dim, output_dim=output_dim, h1=h1, h2=h2, dropout=dropout).to(device)
# # #     criterion = nn.MSELoss()

# # #     # Seleção do otimizador
# # #     if optimizer_name == "adam":
# # #         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# # #     elif optimizer_name == "sgd":
# # #         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# # #     else:
# # #         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

# # #     # Preparar dataset
# # #     X_tensor = torch.tensor(X.values, dtype=torch.float32)
# # #     y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
# # #     dataset = TensorDataset(X_tensor, y_tensor)

# # #     val_size = int(len(dataset) * val_split)
# # #     train_size = len(dataset) - val_size
# # #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # #     # Treinamento
# # #     for epoch in range(n_epochs):
# # #         model.train()
# # #         for batch_X, batch_y in train_loader:
# # #             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
# # #             optimizer.zero_grad()
# # #             preds = model(batch_X)
# # #             loss = criterion(preds, batch_y)
# # #             loss.backward()
# # #             optimizer.step()

# # #     # Avaliação na validação
# # #     model.eval()
# # #     val_loss = 0
# # #     with torch.no_grad():
# # #         for batch_X, batch_y in val_loader:
# # #             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
# # #             preds = model(batch_X)
# # #             val_loss += criterion(preds, batch_y).item() * batch_X.size(0)
# # #     val_loss /= len(val_loader.dataset)

# # #     return val_loss

# # # # --- Função para otimização ---
# # # def optimize_hyperparams(X, y, n_trials=20, n_epochs=50):
# # #     logger.info("Iniciando otimização de hiperparâmetros com Optuna...")
# # #     study = optuna.create_study(direction="minimize")
# # #     study.optimize(lambda trial: objective(trial, X, y, n_epochs=n_epochs), n_trials=n_trials)
# # #     best_params = study.best_trial.params
# # #     logger.info(f"✅ Melhores hiperparâmetros encontrados: {best_params}")
# # #     return best_params

# # import optuna
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, TensorDataset
# # from rna_workflow_engine.get_model import MLP

# # def objective(trial, X_train, y_train):
# #     # Hiperparâmetros a otimizar
# #     h1 = trial.suggest_int("h1", 8, 128)
# #     h2 = trial.suggest_int("h2", 8, 128)
# #     dropout = trial.suggest_float("dropout", 0.0, 0.5)
# #     lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
# #     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
# #     optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

# #     # Dataset
# #     dataset = TensorDataset(
# #         torch.tensor(X_train.values, dtype=torch.float32),
# #         torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# #     )
# #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# #     # Modelo
# #     model = MLP(input_dim=X_train.shape[1], output_dim=1, h1=h1, h2=h2, dropout=dropout)
# #     criterion = nn.MSELoss()
# #     optim_map = {'adam': optim.Adam, 'rmsprop': optim.RMSprop, 'sgd': optim.SGD}
# #     optimizer_cls = optim_map.get(optimizer_name.lower(), optim.Adam)
# #     optimizer = optimizer_cls(model.parameters(), lr=lr)

# #     # Treino rápido para avaliar
# #     model.train()
# #     for xb, yb in loader:
# #         optimizer.zero_grad()
# #         preds = model(xb)
# #         loss = criterion(preds, yb)
# #         loss.backward()
# #         optimizer.step()

# #     # Avaliação no mesmo conjunto (simples)
# #     model.eval()
# #     with torch.no_grad():
# #         X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# #         y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# #         preds = model(X_tensor)
# #         val_loss = criterion(preds, y_tensor).item()

# #     return val_loss

# # def optimize_hyperparams(X_train, y_train, n_trials=20):
# #     """Executa otimização de hiperparâmetros via Optuna"""
# #     study = optuna.create_study(direction="minimize")
# #     study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
# #     return study.best_params

# # # hyperparam_optimization.py
# # import typer
# # from loguru import logger
# # from sklearn.model_selection import train_test_split
# # import optuna
# # from rna_workflow_engine.modeling.train import train_model

# # app = typer.Typer()

# # def objective(trial, X_train, y_train):
# #     params = {
# #         "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
# #         "dropout": trial.suggest_float("dropout", 0.0, 0.5),
# #         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
# #     }

# #     # Treina por poucas epochs para otimização rápida
# #     model, metrics = train_model(X_train, y_train, params=params, epochs=10, batch_size=32)
# #     val_acc = metrics.get("val_accuracy", 0)
# #     return val_acc

# # def optimize_hyperparams(X_train, y_train, n_trials: int = 20, batch_size: int = 32):
# #     """
# #     Executa a otimização de hiperparâmetros usando Optuna.
# #     Retorna os melhores parâmetros encontrados.
# #     """
# #     import optuna
# #     study = optuna.create_study(direction="maximize")
# #     func = lambda trial: objective(trial, X_train, y_train)
# #     study.optimize(func, n_trials=n_trials)
# #     logger.info(f"Melhores parâmetros: {study.best_params}")
# #     return study.best_params

# # @app.command()
# # def run_optimization(
# #     dataset_name: str = typer.Option(..., help="Nome do dataset"),
# #     n_trials: int = typer.Option(20, help="Número de trials"),
# # ):
# #     typer.echo(f"Iniciando otimização de hiperparâmetros para {dataset_name}...")

# # if __name__ == "__main__":
# #     app()

# # rna_workflow_engine/modeling/hyperparam_optimization.py
# import typer
# from loguru import logger
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import optuna
# import torch

# from rna_workflow_engine.modeling.train import train_model
# from rna_workflow_engine.get_model import get_model  # opcional, caso use diretamente o model builder

# app = typer.Typer()

# def objective(trial, X, y):
#     """
#     Função objetivo para o Optuna — testa combinações de hiperparâmetros e retorna a acurácia de validação.
#     """
#     params = {
#         "hidden1": trial.suggest_int("hidden1", 32, 128),
#         "hidden2": trial.suggest_int("hidden2", 16, 64),
#         "dropout": trial.suggest_float("dropout", 0.0, 0.5),
#         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
#     }

#     # Divide em treino e validação
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     # Treina modelo com poucos epochs (para otimização rápida)
#     model, train_losses, val_losses = train_model(
#         X_train=X_train,
#         y_train=y_train,
#         X_val=X_val,
#         y_val=y_val,
#         input_dim=X_train.shape[1],
#         output_dim=len(y.unique()) if len(y.unique()) > 2 else 1,
#         task="classification",
#         hidden1=params["hidden1"],
#         hidden2=params["hidden2"],
#         dropout=params["dropout"],
#         lr=params["lr"],
#         epochs=10,
#         batch_size=32,
#         device="cpu"
#     )

#     # Avaliação rápida
#     model.eval()
#     with torch.no_grad():
#         X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
#         logits = model(X_val_tensor)
#         if len(y.unique()) == 2:
#             preds = (torch.sigmoid(logits).squeeze() >= 0.5).numpy()
#         else:
#             preds = torch.argmax(logits, axis=1).numpy()

#     val_acc = accuracy_score(y_val, preds)
#     logger.info(f"Trial concluído — Val Accuracy: {val_acc:.4f} | Params: {params}")

#     return val_acc


# def optimize_hyperparams(X, y, n_trials: int = 20):
#     """
#     Executa a otimização de hiperparâmetros usando Optuna e retorna os melhores parâmetros encontrados.
#     """
#     logger.info("Iniciando otimização de hiperparâmetros com Optuna...")
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
#     logger.success(f"Melhores parâmetros encontrados: {study.best_params}")
#     return study.best_params


# @app.command()
# def run_optimization(
#     dataset_name: str = typer.Option(..., help="Nome do dataset (ex: iris, titanic, wine, diabetes)"),
#     n_trials: int = typer.Option(20, help="Número de tentativas (trials) de otimização")
# ):
#     """
#     CLI Typer: executa a otimização de hiperparâmetros diretamente pelo terminal.
#     """
#     from rna_workflow_engine.data.dataset import fetch_dataset
#     X, y = fetch_dataset(dataset_name)
#     best_params = optimize_hyperparams(X, y, n_trials=n_trials)
#     typer.echo(f"Melhores parâmetros encontrados para {dataset_name}: {best_params}")


# if __name__ == "__main__":
#     app()


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
