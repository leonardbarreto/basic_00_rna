# import optuna
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from rna_workflow_engine.get_model import get_model
# from rna_workflow_engine.utils import get_device  # função auxiliar para CPU/GPU
# import logging

# logger = logging.getLogger(__name__)

# def optimize_hyperparams(X, y, n_trials=20):
#     """
#     Otimiza hiperparâmetros de um MLP usando Optuna com CV.
#     """
#     device = get_device()

#     def objective(trial):
#         # Define hiperparâmetros
#         h1 = trial.suggest_int("h1", 16, 128)
#         h2 = trial.suggest_int("h2", 16, 128)
#         dropout = trial.suggest_float("dropout", 0.0, 0.5)
#         lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

#         # K-Fold CV
#         kf = KFold(n_splits=3, shuffle=True, random_state=42)
#         cv_loss = 0.0

#         for train_idx, val_idx in kf.split(X):
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]

#             model = get_model(
#                 "mlp", input_dim=X.shape[1], output_dim=1, h1=h1, h2=h2, dropout=dropout
#             ).to(device)

#             optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             criterion = torch.nn.MSELoss()

#             train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                           torch.tensor(y_train, dtype=torch.float32).view(-1,1))
#             val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
#                                         torch.tensor(y_val, dtype=torch.float32).view(-1,1))
#             train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size=32)

#             # Treinamento simples
#             for epoch in range(20):
#                 model.train()
#                 for xb, yb in train_loader:
#                     xb, yb = xb.to(device), yb.to(device)
#                     optimizer.zero_grad()
#                     pred = model(xb)
#                     loss = criterion(pred, yb)
#                     loss.backward()
#                     optimizer.step()

#             # Avalia
#             model.eval()
#             val_preds = []
#             val_targets = []
#             with torch.no_grad():
#                 for xb, yb in val_loader:
#                     xb, yb = xb.to(device), yb.to(device)
#                     val_preds.append(model(xb))
#                     val_targets.append(yb)
#             val_preds = torch.cat(val_preds)
#             val_targets = torch.cat(val_targets)
#             cv_loss += mean_squared_error(val_targets.cpu().numpy(), val_preds.cpu().numpy())

#         return cv_loss / 3

#     study = optuna.create_study(direction="minimize", study_name="MLP_Hyperparam_Optimization")
#     study.optimize(objective, n_trials=n_trials)
#     logger.info(f"Melhores hiperparâmetros: {study.best_params}")
#     return study.best_params

# --- hyperparam_optimization.py ---
import optuna
import torch
from loguru import logger
from rna_workflow_engine.get_model import MLP
from rna_workflow_engine.utils import get_device

def objective(trial, X, y):
    # Espaço de busca de hiperparâmetros
    h1 = trial.suggest_int("h1", 16, 128)
    h2 = trial.suggest_int("h2", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    device = get_device()
    input_dim = X.shape[1]
    output_dim = 1  # ajuste se classificação

    model = MLP(input_dim=input_dim, output_dim=output_dim, h1=h1, h2=h2, dropout=dropout).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataset simples (todo em memória)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = criterion(preds, y_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()


def optimize_hyperparams(X, y, n_trials=20):
    logger.info("Iniciando otimização de hiperparâmetros com Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    best_params = study.best_trial.params
    logger.info(f"Melhores hiperparâmetros: {best_params}")
    return best_params
