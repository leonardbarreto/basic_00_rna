# # import os

# # import torch
# # from loguru import logger

# # from rna_workflow_engine.config import REPORTS_DIR
# # from rna_workflow_engine.dataset import fetch_dataset
# # from rna_workflow_engine.get_model import get_model


# # # def evaluate_model(model_path: str, dataset_name: str, h1: int, h2: int, dropout: float):
# # #     X, y = fetch_dataset(dataset_name)
# # #     input_dim = X.shape[1]
# # #     output_dim = 1
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # #     # --- Instancia MLP com mesmos hiperparâmetros do treino ---
# # #     model = get_model('mlp', input_dim=input_dim, output_dim=output_dim)
# # #     model = model.to(device)
# # #     model.eval()

# # #     state_dict = torch.load(model_path, map_location=device)
# # #     model.load_state_dict(state_dict)

# # #     X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
# # #     with torch.no_grad():
# # #         preds = model(X_tensor).cpu().numpy()

# # #     mse = ((preds.flatten() - y.values) ** 2).mean()
# # #     logger.info(f"Avaliação final - MSE: {mse:.4f}")

# # #     return {"mse": mse}

# # # import torch

# # # def evaluate_model(model_class, state_dict_path, best_params, X_test, y_test, device='cpu'):
# # #     """
# # #     Avalia um modelo treinado.

# # #     Parâmetros:
# # #     - model_class: a classe do modelo (ex.: MLP)
# # #     - state_dict_path: caminho para o arquivo .pt ou .pth do modelo salvo
# # #     - best_params: dicionário com os melhores hiperparâmetros usados no treino (ex.: h1, h2, dropout)
# # #     - X_test: tensor ou array dos dados de teste
# # #     - y_test: tensor ou array dos targets de teste
# # #     - device: 'cpu' ou 'cuda'

# # #     Retorna:
# # #     - loss e predições do modelo
# # #     """

# # #     # Instancia o modelo com os mesmos hiperparâmetros usados no treino
# # #     model = model_class(
# # #         h1=best_params['h1'],
# # #         h2=best_params['h2'],
# # #         input_dim=X_test.shape[1],
# # #         output_dim=1,
# # #         dropout=best_params.get('dropout', 0.0)
# # #     ).to(device)

# # #     # Carrega os pesos salvos
# # #     model.load_state_dict(torch.load(state_dict_path, map_location=device))
# # #     model.eval()

# # #     # Avaliação
# # #     with torch.no_grad():
# # #         X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# # #         y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
# # #         predictions = model(X_tensor)
# # #         loss_fn = torch.nn.MSELoss()
# # #         loss = loss_fn(predictions.squeeze(), y_tensor)

# # #     return loss.item(), predictions.cpu().numpy()

# # import torch
# # import torch.nn as nn
# # from sklearn.metrics import mean_squared_error, mean_absolute_error
# # import mlflow
# # import joblib
# # import os

# # def evaluate_model(model, X_test, y_test, criterion=None, device='cpu'):
# #     """
# #     Avalia um modelo treinado no conjunto de teste.
    
# #     Args:
# #         model (torch.nn.Module): modelo treinado.
# #         X_test (np.array ou torch.Tensor): features de teste.
# #         y_test (np.array ou torch.Tensor): targets de teste.
# #         criterion (torch loss, optional): função de perda (default: MSELoss).
# #         device (str): 'cpu' ou 'cuda'.
        
# #     Returns:
# #         dict: métricas de avaliação {'mse', 'mae'}
# #     """
# #     if criterion is None:
# #         criterion = nn.MSELoss()
    
# #     model.to(device)
# #     model.eval()
    
# #     with torch.no_grad():
# #         X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# #         y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
# #         y_pred = model(X_test_tensor)
# #         mse = criterion(y_pred, y_test_tensor).item()
# #         mae = mean_absolute_error(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())
    
# #     # Log das métricas no MLflow
# #     mlflow.log_metric("test_mse", mse)
# #     mlflow.log_metric("test_mae", mae)
    
# #     return {'mse': mse, 'mae': mae}


# # def load_model(model_class, model_path, device='cpu', **model_kwargs):
# #     """
# #     Carrega modelo salvo (PyTorch).
    
# #     Args:
# #         model_class: classe do modelo.
# #         model_path (str): caminho do arquivo .pt ou .pth.
# #         device (str): 'cpu' ou 'cuda'.
# #         model_kwargs: argumentos do construtor do modelo.
        
# #     Returns:
# #         modelo carregado.
# #     """
# #     model = model_class(**model_kwargs).to(device)
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model.eval()
# #     return model


# # def evaluate_model_from_path(model_class, model_path, X_test, y_test, device='cpu', **model_kwargs):
# #     """
# #     Avalia modelo carregado do caminho (combina load_model + evaluate_model).
    
# #     Args:
# #         model_class: classe do modelo.
# #         model_path: caminho do modelo salvo.
# #         X_test, y_test: dados de teste.
# #         device: cpu ou cuda
# #         model_kwargs: parâmetros do modelo
    
# #     Returns:
# #         dict de métricas
# #     """
# #     model = load_model(model_class, model_path, device=device, **model_kwargs)
# #     metrics = evaluate_model(model, X_test, y_test, device=device)
# #     return metrics

# # --- evaluator.py ---
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from loguru import logger

# def evaluate_model(model_path, dataset_name='iris', h1=None, h2=None, dropout=None):
#     """
#     Avalia o modelo salvo no conjunto de teste.
    
#     Parâmetros:
#     - model_path: str, caminho do modelo Keras salvo
#     - dataset_name: str, nome do dataset ('iris')
#     - h1, h2, dropout: hiperparâmetros (não usados no momento, mas mantidos para compatibilidade)
    
#     Retorna:
#     - metrics: dict com MSE, RMSE, MAE
#     """
#     # --- Carregar dataset ---
#     if dataset_name == 'iris':
#         data = load_iris()
#         X, y = data.data, data.target
#         y = y.astype(float)
#     else:
#         raise ValueError("Dataset não suportado")

#     # --- Split teste final (mesmo do treino) ---
#     _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # --- Carregar modelo ---
#     model = load_model(model_path)

#     # --- Previsões ---
#     y_pred = model.predict(X_test)

#     # --- Métricas ---
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred)

#     metrics = {
#         "mse": float(mse),
#         "rmse": float(rmse),
#         "mae": float(mae)
#     }

#     logger.info(f"📊 Métricas de Teste: {metrics}")
#     return metrics

# --- evaluator.py ---
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from rna_workflow_engine.get_model import get_model


def evaluate_model(model_path: str, X_test: pd.DataFrame, y_test: pd.DataFrame,
                   h1: int, h2: int, dropout: float):
    """
    Avalia um modelo PyTorch MLP salvo no disco.

    Parâmetros:
    ----------
    model_path : str
        Caminho do arquivo .pt ou .pth com state_dict do modelo.
    X_test : pd.DataFrame
        Dados de entrada de teste.
    y_test : pd.DataFrame
        Valores reais de saída de teste.
    h1, h2 : int
        Número de neurônios nas camadas ocultas.
    dropout : float
        Taxa de dropout.

    Retorna:
    -------
    metrics : dict
        Dicionário com MSE, MAE e R2.
    """
    # Detecta dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializa modelo
    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1] if len(y_test.shape) > 1 else 1
    model = get_model("mlp", input_dim=input_dim, output_dim=output_dim, task="regression")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dados de teste
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        y_pred = model(X_tensor)

    # Métricas
    y_true_np = y_tensor.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    metrics = {"mse": mse, "mae": mae, "r2": r2}
    return metrics
