# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from rna_workflow_engine.get_model import get_model
from loguru import logger
from typing import Tuple, List

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim: int,
    output_dim: int,
    task: str = "classification",
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Treina um modelo MLP PyTorch.
    """
    device = torch.device(device)
    model = get_model("mlp", input_dim=input_dim, output_dim=output_dim, task=task).to(device)

    criterion = nn.BCEWithLogitsLoss() if task=="classification" and output_dim==1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32 if task=="classification" and output_dim==1 else torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32 if task=="classification" and output_dim==1 else torch.long)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze() if output_dim==1 else outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze() if output_dim==1 else outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    return model, train_losses, val_losses
