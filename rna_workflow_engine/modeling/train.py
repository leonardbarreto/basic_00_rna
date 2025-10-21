# # # # --- train.py ---
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader, TensorDataset
# # # from loguru import logger

# # # from rna_workflow_engine.dataset import fetch_dataset
# # # from rna_workflow_engine.get_model import MLP
# # # from rna_workflow_engine.experiments.mlflow_utils import start_experiment, log_params, log_metrics, log_model
# # # from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams


# # # def get_device():
# # #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # # def train(dataset_name: str = "iris", n_trials: int = 20):
# # #     logger.info(f"Carregando dataset '{dataset_name}'")
# # #     X, y = fetch_dataset(dataset_name)

# # #     # --- Otimização de hiperparâmetros ---
# # #     best_params = optimize_hyperparams(dataset_name, n_trials=n_trials)
# # #     logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

# # #     device = get_device()
# # #     logger.info(f"Usando dispositivo: {device}")

# # #     input_dim = X.shape[1]
# # #     output_dim = 1  # para regressão simples, ajustar para classificação se necessário

# # #     # --- Modelo ---
# # #     model = MLP(
# # #         input_dim=input_dim,
# # #         output_dim=output_dim,
# # #         h1=best_params['h1'],
# # #         h2=best_params['h2'],
# # #         dropout=best_params['dropout']
# # #     ).to(device)

# # #     criterion = nn.MSELoss()
# # #     optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

# # #     # --- Cross-validation simples (3 folds) ---
# # #     fold_size = len(X) // 3
# # #     indices = np.arange(len(X))

# # #     start_experiment("MLP_Regression")
# # #     log_params(best_params)

# # #     metrics_list = []
# # #     for fold in range(3):
# # #         val_idx = indices[fold*fold_size:(fold+1)*fold_size]
# # #         train_idx = np.setdiff1d(indices, val_idx)

# # #         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
# # #         X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

# # #         train_loader = DataLoader(TensorDataset(
# # #             torch.tensor(X_train.values, dtype=torch.float32),
# # #             torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# # #         ), batch_size=best_params.get('batch_size', 32), shuffle=True)

# # #         val_loader = DataLoader(TensorDataset(
# # #             torch.tensor(X_val.values, dtype=torch.float32),
# # #             torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
# # #         ), batch_size=best_params.get('batch_size', 32), shuffle=False)

# # #         # Treinamento
# # #         model.train()
# # #         train_losses = []
# # #         for xb, yb in train_loader:
# # #             xb, yb = xb.to(device), yb.to(device)
# # #             optimizer.zero_grad()
# # #             preds = model(xb)
# # #             loss = criterion(preds, yb)
# # #             loss.backward()
# # #             optimizer.step()
# # #             train_losses.append(loss.item())

# # #         # Validação
# # #         model.eval()
# # #         val_losses = []
# # #         with torch.no_grad():
# # #             for xb, yb in val_loader:
# # #                 xb, yb = xb.to(device), yb.to(device)
# # #                 preds = model(xb)
# # #                 val_losses.append(criterion(preds, yb).item())

# # #         fold_train_loss = np.mean(train_losses)
# # #         fold_val_loss = np.mean(val_losses)
# # #         metrics_list.append(
# # #             {'fold': fold+1, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss})
# # #         log_metrics({'fold_train_loss': fold_train_loss,
# # #                      'fold_val_loss': fold_val_loss})
# # #         logger.info(
# # #             f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}")

# # #     # --- Salvar modelo final ---
# # #     model_path = "models/best_model.pt"
# # #     torch.save(model.state_dict(), model_path)
# # #     log_model(model)

# # #     avg_val_loss = np.mean([m['val_loss'] for m in metrics_list])
# # #     logger.success(
# # #         f"Treinamento concluído. Val Loss médio: {avg_val_loss:.4f}")

# # #     return model, best_params, metrics_list
# # # --- train.py ---
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, TensorDataset
# # from loguru import logger

# # from rna_workflow_engine.dataset import fetch_dataset
# # from rna_workflow_engine.get_model import MLP
# # from rna_workflow_engine.experiments.mlflow_utils import start_experiment, log_params, log_metrics, log_model
# # from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams


# # def get_device():
# #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # def train(dataset_name: str = "iris", n_trials: int = 20):
# #     logger.info(f"Carregando dataset '{dataset_name}'")
# #     X, y = fetch_dataset(dataset_name)

# #     # --- Otimização de hiperparâmetros ---
# #     best_params = optimize_hyperparams(X, y, n_trials=n_trials)  # ✅ corrigido
# #     logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

# #     device = get_device()
# #     logger.info(f"Usando dispositivo: {device}")

# #     input_dim = X.shape[1]
# #     output_dim = 1  # para regressão simples, ajustar para classificação se necessário

# #     # --- Modelo ---
# #     model = MLP(
# #         input_dim=input_dim,
# #         output_dim=output_dim,
# #         h1=best_params['h1'],
# #         h2=best_params['h2'],
# #         dropout=best_params['dropout']
# #     ).to(device)

# #     criterion = nn.MSELoss()
# #     optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

# #     # --- Cross-validation simples (3 folds) ---
# #     fold_size = len(X) // 3
# #     indices = np.arange(len(X))

# #     start_experiment("MLP_Regression")
# #     log_params(best_params)

# #     metrics_list = []
# #     for fold in range(3):
# #         val_idx = indices[fold*fold_size:(fold+1)*fold_size]
# #         train_idx = np.setdiff1d(indices, val_idx)

# #         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
# #         X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

# #         train_loader = DataLoader(TensorDataset(
# #             torch.tensor(X_train.values, dtype=torch.float32),
# #             torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# #         ), batch_size=best_params.get('batch_size', 32), shuffle=True)

# #         val_loader = DataLoader(TensorDataset(
# #             torch.tensor(X_val.values, dtype=torch.float32),
# #             torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
# #         ), batch_size=best_params.get('batch_size', 32), shuffle=False)

# #         # Treinamento
# #         model.train()
# #         train_losses = []
# #         for xb, yb in train_loader:
# #             xb, yb = xb.to(device), yb.to(device)
# #             optimizer.zero_grad()
# #             preds = model(xb)
# #             loss = criterion(preds, yb)
# #             loss.backward()
# #             optimizer.step()
# #             train_losses.append(loss.item())

# #         # Validação
# #         model.eval()
# #         val_losses = []
# #         with torch.no_grad():
# #             for xb, yb in val_loader:
# #                 xb, yb = xb.to(device), yb.to(device)
# #                 preds = model(xb)
# #                 val_losses.append(criterion(preds, yb).item())

# #         fold_train_loss = np.mean(train_losses)
# #         fold_val_loss = np.mean(val_losses)
# #         metrics_list.append(
# #             {'fold': fold+1, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss})
# #         log_metrics({'fold_train_loss': fold_train_loss,
# #                      'fold_val_loss': fold_val_loss})
# #         logger.info(
# #             f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}")

# #     # --- Salvar modelo final ---
# #     #model_path = "models/best_model.pt"
# #     #torch.save(model.state_dict(), model_path)
# #     log_model(model)

# #     avg_val_loss = np.mean([m['val_loss'] for m in metrics_list])
# #     logger.success(
# #         f"Treinamento concluído. Val Loss médio: {avg_val_loss:.4f}")

# #     return model, best_params, metrics_list
# # --- train.py ---
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from loguru import logger
# import matplotlib.pyplot as plt

# from rna_workflow_engine.dataset import fetch_dataset
# from rna_workflow_engine.get_model import MLP
# from rna_workflow_engine.experiments.mlflow_utils import (
#     start_experiment, log_params, log_metrics, log_model, log_figure
# )
# from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams
# from rna_workflow_engine.plots import plot_training_curves_summary


# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train(dataset_name: str = "iris", n_trials: int = 20, n_folds: int = 3):
#     logger.info(f"Carregando dataset '{dataset_name}'")
#     X, y = fetch_dataset(dataset_name)

#     # --- Otimização de hiperparâmetros ---
#     best_params = optimize_hyperparams(X, y, n_trials=n_trials)
#     logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

#     device = get_device()
#     logger.info(f"Usando dispositivo: {device}")

#     input_dim = X.shape[1]
#     output_dim = 1  # para regressão simples, ajustar para classificação se necessário

#     # --- Modelo ---
#     model = MLP(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         h1=best_params['h1'],
#         h2=best_params['h2'],
#         dropout=best_params['dropout']
#     ).to(device)

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

#     # --- Cross-validation ---
#     fold_size = len(X) // n_folds
#     indices = np.arange(len(X))

#     start_experiment("MLP_Regression")
#     log_params(best_params)

#     metrics_list = []
#     for fold in range(n_folds):
#         val_idx = indices[fold*fold_size:(fold+1)*fold_size]
#         train_idx = np.setdiff1d(indices, val_idx)

#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

#         train_loader = DataLoader(TensorDataset(
#             torch.tensor(X_train.values, dtype=torch.float32),
#             torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
#         ), batch_size=best_params.get('batch_size', 32), shuffle=True)

#         val_loader = DataLoader(TensorDataset(
#             torch.tensor(X_val.values, dtype=torch.float32),
#             torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
#         ), batch_size=best_params.get('batch_size', 32), shuffle=False)

#         # Treinamento
#         model.train()
#         train_losses = []
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             train_losses.append(loss.item())

#         # Validação
#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 preds = model(xb)
#                 val_losses.append(criterion(preds, yb).item())

#         fold_train_loss = np.mean(train_losses)
#         fold_val_loss = np.mean(val_losses)
#         metrics_list.append(
#             {'fold': fold+1, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss}
#         )
#         log_metrics({
#             f'fold_{fold+1}_train_loss': fold_train_loss,
#             f'fold_{fold+1}_val_loss': fold_val_loss
#         })
#         logger.info(
#             f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}"
#         )

#     # --- Logar modelo final no MLflow ---
#     log_model(model)

#     # --- Gráfico consolidado dos folds ---
#     fig_summary = plot_training_curves_summary(metrics_list)
#     log_figure(fig_summary, "training_curves_summary.png")
#     plt.close(fig_summary)

#     avg_val_loss = np.mean([m['val_loss'] for m in metrics_list])
#     logger.success(f"Treinamento concluído. Val Loss médio: {avg_val_loss:.4f}")

#     return model, best_params, metrics_list
# --- train.py ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import os

from rna_workflow_engine.dataset import fetch_dataset
from rna_workflow_engine.get_model import MLP
from rna_workflow_engine.experiments.mlflow_utils import start_experiment, log_params, log_metrics, log_model, log_figure
from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams
from rna_workflow_engine.plots import plot_training_validation
from rna_workflow_engine.config import REPORTS_DIR  # caminho dos relatórios


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_name: str = "iris", n_trials: int = 20, n_folds: int = 3):
    logger.info(f"Carregando dataset '{dataset_name}'")
    X, y = fetch_dataset(dataset_name)

    # --- Otimização de hiperparâmetros ---
    best_params = optimize_hyperparams(X, y, n_trials=n_trials)
    logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

    device = get_device()
    logger.info(f"Usando dispositivo: {device}")

    input_dim = X.shape[1]
    output_dim = 1  # ajustar para classificação se necessário

    # --- Modelo ---
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        h1=best_params['h1'],
        h2=best_params['h2'],
        dropout=best_params['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    # --- Cross-validation ---
    fold_size = len(X) // n_folds
    indices = np.arange(len(X))

    start_experiment("MLP_Regression")
    log_params(best_params)

    metrics_list = []
    for fold in range(n_folds):
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        ), batch_size=best_params.get('batch_size', 32), shuffle=True)

        val_loader = DataLoader(TensorDataset(
            torch.tensor(X_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        ), batch_size=best_params.get('batch_size', 32), shuffle=False)

        # --- Treinamento ---
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validação ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())

        fold_train_loss = np.mean(train_losses)
        fold_val_loss = np.mean(val_losses)
        metrics_list.append({'fold': fold+1, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss})
        log_metrics({'fold_train_loss': fold_train_loss, 'fold_val_loss': fold_val_loss})
        logger.info(f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}")

        # --- Salvar gráfico de treinamento/validação por fold ---
        fig = plot_training_validation(train_losses, val_losses, fold+1)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig_path = os.path.join(REPORTS_DIR, f"training_validation_fold{fold+1}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        log_figure(fig, f"training_validation_fold{fold+1}.png")
        fig.clf()  # limpa figura para liberar memória

    avg_val_loss = np.mean([m['val_loss'] for m in metrics_list])
    logger.success(f"Treinamento concluído. Val Loss médio: {avg_val_loss:.4f}")

    # --- Logar modelo final no MLflow ---
    log_model(model)

    return model, best_params, metrics_list
