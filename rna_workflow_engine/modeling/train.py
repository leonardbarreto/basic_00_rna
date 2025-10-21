# # import os

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from loguru import logger
# # from torch.utils.data import DataLoader, TensorDataset

# # from rna_workflow_engine.config import REPORTS_DIR  # caminho dos relatórios
# # from rna_workflow_engine.dataset import fetch_dataset
# # from rna_workflow_engine.experiments.mlflow_utils import (log_figure,
# #                                                           log_metrics,
# #                                                           log_model,
# #                                                           log_params,
# #                                                           start_experiment)
# # from rna_workflow_engine.get_model import MLP
# # from rna_workflow_engine.modeling.hyperparam_optimization import \
# #     optimize_hyperparams
# # from rna_workflow_engine.plots import plot_training_validation


# # def get_device():
# #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # def train(
# #     dataset_name: str = "iris",
# #     n_trials: int = 20,
# #     n_folds: int = 3,
# #     test_split: float = 0.2,
# # ):
# #     """
# #     Treina um MLP com otimização de hiperparâmetros via Optuna,
# #     realiza cross-validation e avalia no conjunto de teste final.
# #     Tudo logado em um único run MLflow.
# #     """
# #     logger.info(f"Carregando dataset '{dataset_name}'")
# #     X, y = fetch_dataset(dataset_name)

# #     # --- Separar teste ---
# #     n_test = int(len(X) * test_split)
# #     X_trainval, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
# #     y_trainval, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

# #     # --- Otimização de hiperparâmetros ---
# #     best_params = optimize_hyperparams(
# #         X_trainval, y_trainval, n_trials=n_trials)
# #     logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

# #     device = get_device()
# #     logger.info(f"Usando dispositivo: {device}")

# #     input_dim = X.shape[1]
# #     output_dim = 1  # ajustar para classificação se necessário

# #     # --- Modelo ---
# #     model = MLP(
# #         input_dim=input_dim,
# #         output_dim=output_dim,
# #         h1=best_params['h1'],
# #         h2=best_params['h2'],
# #         dropout=best_params['dropout'],
# #     ).to(device)

# #     # --- Otimizador robusto ---
# #     optim_map = {
# #         'adam': optim.Adam,
# #         'rmsprop': optim.RMSprop,
# #         'sgd': optim.SGD,
# #     }
# #     optimizer_cls = optim_map.get(best_params.get(
# #         'optimizer', 'adam').lower(), optim.Adam)
# #     optimizer = optimizer_cls(model.parameters(), lr=best_params['lr'])

# #     criterion = nn.MSELoss()

# #     # --- Cross-validation ---
# #     fold_size = len(X_trainval) // n_folds
# #     indices = np.arange(len(X_trainval))

# #     # --- Iniciar run MLflow ---
# #     run_name = f"{dataset_name}_MLP_Optuna"
# #     run = start_experiment("MLP_Regression", run_name=run_name)
# #     log_params(best_params)

# #     metrics_list = []

# #     for fold in range(n_folds):
# #         val_idx = indices[fold*fold_size:(fold+1)*fold_size]
# #         train_idx = np.setdiff1d(indices, val_idx)

# #         X_train, y_train = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
# #         X_val, y_val = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

# #         train_loader = DataLoader(
# #             TensorDataset(
# #                 torch.tensor(X_train.values, dtype=torch.float32),
# #                 torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# #             ),
# #             batch_size=best_params.get('batch_size', 32),
# #             shuffle=True
# #         )
# #         val_loader = DataLoader(
# #             TensorDataset(
# #                 torch.tensor(X_val.values, dtype=torch.float32),
# #                 torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
# #             ),
# #             batch_size=best_params.get('batch_size', 32),
# #             shuffle=False
# #         )

# #         # --- Treinamento ---
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

# #         # --- Validação ---
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
# #                     'fold_val_loss': fold_val_loss})
# #         logger.info(
# #             f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}")

# #         # --- Salvar gráfico ---
# #         fig = plot_training_validation(train_losses, val_losses, fold+1)
# #         os.makedirs(REPORTS_DIR, exist_ok=True)
# #         fig_path = os.path.join(
# #             REPORTS_DIR, f"training_validation_fold{fold+1}.png")
# #         fig.savefig(fig_path, bbox_inches='tight')
# #         log_figure(fig, f"training_validation_fold{fold+1}.png")
# #         fig.clf()

# #     # --- Avaliação Final (Teste) ---
# #     test_loader = DataLoader(
# #         TensorDataset(
# #             torch.tensor(X_test.values, dtype=torch.float32),
# #             torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
# #         ),
# #         batch_size=best_params.get('batch_size', 32),
# #         shuffle=False
# #     )

# #     model.eval()
# #     test_losses = []
# #     with torch.no_grad():
# #         for xb, yb in test_loader:
# #             xb, yb = xb.to(device), yb.to(device)
# #             preds = model(xb)
# #             test_losses.append(criterion(preds, yb).item())

# #     test_mse = np.mean(test_losses)
# #     log_metrics({'test_mse': test_mse})
# #     logger.success(
# #         f"✅ Pipeline completo: treino, validação e teste no mesmo run! Test MSE: {test_mse:.4f}")

# #     # --- Logar modelo final ---
# #     model_path = os.path.join(REPORTS_DIR, f"{dataset_name}_mlp_final.pt")
# #     torch.save(model.state_dict(), model_path)
# #     log_model(model)

# #     return model_path, best_params, metrics_list
# # --- train.py (PyTorch + MLflow + Dataset Logging) ---
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from loguru import logger
# import mlflow
# from mlflow.models.signature import infer_signature
# import pandas as pd
# import tempfile

# from rna_workflow_engine.config import REPORTS_DIR
# from rna_workflow_engine.dataset import fetch_dataset
# from rna_workflow_engine.experiments.mlflow_utils import log_figure, log_metrics, log_model, log_params, start_experiment
# from rna_workflow_engine.get_model import MLP
# from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams
# from rna_workflow_engine.plots import plot_training_validation

# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def train(dataset_name: str = "iris", n_trials: int = 20, n_folds: int = 3, test_split: float = 0.2):
#     """
#     Treina um MLP com otimização de hiperparâmetros via Optuna,
#     realiza cross-validation e avalia no conjunto de teste final.
#     Loga dataset, métricas, figuras e modelo no MLflow.
#     """
#     logger.info(f"Carregando dataset '{dataset_name}'")
#     X, y = fetch_dataset(dataset_name)

#     # --- Separar teste ---
#     n_test = int(len(X) * test_split)
#     X_trainval, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
#     y_trainval, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

#     # --- Log dataset no MLflow ---
#     dataset_df = pd.concat([X_trainval, y_trainval], axis=1)
#     dataset_df.columns = list(X_trainval.columns) + ["target"]
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
#         dataset_df.to_csv(tmp_file.name, index=False)
#         mlflow.log_artifact(tmp_file.name, artifact_path="datasets")
#         logger.info(f"Dataset logado no MLflow em {tmp_file.name}")

#     # --- Otimização de hiperparâmetros ---
#     best_params = optimize_hyperparams(X_trainval, y_trainval, n_trials=n_trials)
#     logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

#     device = get_device()
#     logger.info(f"Usando dispositivo: {device}")

#     input_dim = X.shape[1]
#     output_dim = 1

#     # --- Modelo ---
#     model = MLP(input_dim=input_dim, output_dim=output_dim,
#                 h1=best_params['h1'], h2=best_params['h2'], dropout=best_params['dropout']).to(device)

#     optim_map = {'adam': optim.Adam, 'rmsprop': optim.RMSprop, 'sgd': optim.SGD}
#     optimizer_cls = optim_map.get(best_params.get('optimizer', 'adam').lower(), optim.Adam)
#     optimizer = optimizer_cls(model.parameters(), lr=best_params['lr'])
#     criterion = nn.MSELoss()

#     # --- Cross-validation ---
#     fold_size = len(X_trainval) // n_folds
#     indices = np.arange(len(X_trainval))
#     metrics_list = []

#     run_name = f"{dataset_name}_MLP_Optuna"
#     run = start_experiment("MLP_Regression", run_name=run_name)
#     log_params(best_params)

#     for fold in range(n_folds):
#         val_idx = indices[fold*fold_size:(fold+1)*fold_size]
#         train_idx = np.setdiff1d(indices, val_idx)

#         X_train, y_train = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
#         X_val, y_val = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

#         train_loader = DataLoader(
#             TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
#                           torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)),
#             batch_size=best_params.get('batch_size', 32), shuffle=True
#         )
#         val_loader = DataLoader(
#             TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
#                           torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)),
#             batch_size=best_params.get('batch_size', 32), shuffle=False
#         )

#         # --- Treinamento ---
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

#         # --- Validação ---
#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 preds = model(xb)
#                 val_losses.append(criterion(preds, yb).item())

#         fold_train_loss = np.mean(train_losses)
#         fold_val_loss = np.mean(val_losses)
#         metrics_list.append({'fold': fold+1, 'train_loss': fold_train_loss, 'val_loss': fold_val_loss})
#         log_metrics({'fold_train_loss': fold_train_loss, 'fold_val_loss': fold_val_loss})
#         logger.info(f"Fold {fold+1} - Train Loss: {fold_train_loss:.4f}, Val Loss: {fold_val_loss:.4f}")

#         # --- Gráfico ---
#         fig = plot_training_validation(train_losses, val_losses, fold+1)
#         os.makedirs(REPORTS_DIR, exist_ok=True)
#         fig_path = os.path.join(REPORTS_DIR, f"training_validation_fold{fold+1}.png")
#         fig.savefig(fig_path, bbox_inches='tight')
#         log_figure(fig, f"training_validation_fold{fold+1}.png")
#         fig.clf()

#     # --- Avaliação Final (Teste) ---
#     test_loader = DataLoader(
#         TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
#                       torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)),
#         batch_size=best_params.get('batch_size', 32), shuffle=False
#     )

#     model.eval()
#     test_preds, test_targets = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds = model(xb)
#             test_preds.append(preds.cpu())
#             test_targets.append(yb.cpu())

#     test_preds = torch.cat(test_preds, dim=0).numpy()
#     test_targets = torch.cat(test_targets, dim=0).numpy()
#     test_mse = np.mean((test_preds - test_targets)**2)
#     log_metrics({'test_mse': test_mse})
#     logger.success(f"✅ Pipeline completo! Test MSE: {test_mse:.4f}")

#     # --- Log modelo com infer_signature ---
#     model_path = os.path.join(REPORTS_DIR, f"{dataset_name}_mlp_final.pt")
#     torch.save(model.state_dict(), model_path)

#     # Criar dummy input para assinatura
#     example_input = torch.tensor(X_trainval.values[:5], dtype=torch.float32)
#     example_output = model(example_input)
#     signature = infer_signature(example_input.numpy(), example_output.detach().numpy())
#     mlflow.pytorch.log_model(model, "model", signature=signature)
#     logger.info(f"Modelo logado no MLflow com signature. Path: {model_path}")

#     return model_path, best_params, metrics_list

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import tempfile

from rna_workflow_engine.config import REPORTS_DIR
from rna_workflow_engine.dataset import fetch_dataset
from rna_workflow_engine.experiments.mlflow_utils import (
    log_figure, log_metrics, log_model, log_params, start_experiment
)
from rna_workflow_engine.get_model import MLP
from rna_workflow_engine.modeling.hyperparam_optimization import optimize_hyperparams
from rna_workflow_engine.plots import plot_training_validation

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataset_name: str = "iris", n_trials: int = 20, n_folds: int = 3, test_split: float = 0.2):
    """
    Treina um MLP com otimização de hiperparâmetros via Optuna,
    realiza cross-validation e avalia no conjunto de teste final.
    Loga dataset, métricas, figuras e modelo no MLflow com nomes das variáveis.
    """
    logger.info(f"Carregando dataset '{dataset_name}'")
    X, y = fetch_dataset(dataset_name)

    # --- Separar teste ---
    n_test = int(len(X) * test_split)
    X_trainval, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_trainval, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    # --- Log dataset no MLflow ---
    dataset_df = pd.concat([X_trainval, y_trainval], axis=1)
    dataset_df.columns = list(X_trainval.columns) + ["target"]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        dataset_df.to_csv(tmp_file.name, index=False)
        mlflow.log_artifact(tmp_file.name, artifact_path="datasets")
        logger.info(f"Dataset logado no MLflow em {tmp_file.name}")

    # --- Otimização de hiperparâmetros ---
    best_params = optimize_hyperparams(X_trainval, y_trainval, n_trials=n_trials)
    logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")

    device = get_device()
    logger.info(f"Usando dispositivo: {device}")

    input_dim = X.shape[1]
    output_dim = 1

    # --- Modelo ---
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        h1=best_params['h1'],
        h2=best_params['h2'],
        dropout=best_params['dropout']
    ).to(device)

    optim_map = {'adam': optim.Adam, 'rmsprop': optim.RMSprop, 'sgd': optim.SGD}
    optimizer_cls = optim_map.get(best_params.get('optimizer', 'adam').lower(), optim.Adam)
    optimizer = optimizer_cls(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    # --- Cross-validation ---
    fold_size = len(X_trainval) // n_folds
    indices = np.arange(len(X_trainval))
    metrics_list = []

    run_name = f"{dataset_name}_MLP_Optuna"
    run = start_experiment("MLP_Regression", run_name=run_name)
    log_params(best_params)

    for fold in range(n_folds):
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val, y_val = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                          torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)),
            batch_size=best_params.get('batch_size', 32), shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                          torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)),
            batch_size=best_params.get('batch_size', 32), shuffle=False
        )

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

        # --- Gráfico ---
        fig = plot_training_validation(train_losses, val_losses, fold+1)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig_path = os.path.join(REPORTS_DIR, f"training_validation_fold{fold+1}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        log_figure(fig, f"training_validation_fold{fold+1}.png")
        fig.clf()

    # --- Avaliação Final (Teste) ---
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                      torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)),
        batch_size=best_params.get('batch_size', 32), shuffle=False
    )

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            test_preds.append(preds.cpu())
            test_targets.append(yb.cpu())

    test_preds = torch.cat(test_preds, dim=0).numpy()
    test_targets = torch.cat(test_targets, dim=0).numpy()
    test_mse = np.mean((test_preds - test_targets)**2)
    log_metrics({'test_mse': test_mse})
    #logger.success(f"✅ Pipeline completo! Test MSE: {test_mse:.4f}")

    # --- Log modelo com signature e nomes das variáveis ---
    model_path = os.path.join(REPORTS_DIR, f"{dataset_name}_mlp_final.pt")
    torch.save(model.state_dict(), model_path)

    # Criar dummy input/output com nomes reais
    example_input_df = X_trainval.head(5).copy()  # mantém nomes das colunas
    example_output = model(torch.tensor(example_input_df.values, dtype=torch.float32)).detach().numpy()
    example_output_df = pd.DataFrame(example_output, columns=[y_trainval.name if hasattr(y_trainval, 'name') else "target"])

    signature = infer_signature(example_input_df, example_output_df)
    mlflow.pytorch.log_model(model, "model", signature=signature)
    logger.info(f"Modelo logado no MLflow com signature e nomes das variáveis. Path: {model_path}")

    return model_path, best_params, metrics_list
