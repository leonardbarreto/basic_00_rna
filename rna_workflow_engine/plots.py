# plots.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import confusion_matrix


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# plots.py


def plot_loss_curve(history: dict, save_path: Path, title: str = "Loss Curve"):
    """
    Plota e salva a curva de loss.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss curve salva em {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str = "Confusion Matrix"):
    """
    Plota e salva a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Matriz de confusão salva em {save_path}")


def get_classification_preds(model, X, output_dim: int):
    """
    Retorna as previsões do modelo para classificação (binária ou multi-classe)
    """
    with torch.no_grad():
        logits = model(torch.tensor(X.values, dtype=torch.float32))
        if output_dim == 1:
            return (torch.sigmoid(logits).squeeze() >= 0.5).numpy()
        else:
            return torch.argmax(logits, axis=1).numpy()
