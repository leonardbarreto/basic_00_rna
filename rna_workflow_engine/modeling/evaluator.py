# evaluator.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

def evaluate_model(model, X, y, task="classification", device="cpu"):
    """
    Avalia modelo e retorna métricas + previsões
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    X_tensor = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred_np = y_pred.cpu().numpy()

    y_true = np.array(y)
    metrics = {}

    if task=="classification":
        if y_pred_np.shape[1] > 1:
            preds = np.argmax(y_pred_np, axis=1)
        else:
            preds = (y_pred_np > 0.5).astype(int).ravel()
        metrics["accuracy"] = accuracy_score(y_true, preds)
        metrics["precision"] = precision_score(y_true, preds, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_true, preds, average="macro", zero_division=0)
        metrics["f1_score"] = f1_score(y_true, preds, average="macro", zero_division=0)
    else:
        metrics["mse"] = mean_squared_error(y_true, y_pred_np)
        metrics["r2"] = r2_score(y_true, y_pred_np)

    return metrics, preds if task=="classification" else y_pred_np
