import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import optuna
from loguru import logger
from rna_workflow_engine.utils import get_device

# --- Exemplo de MLP ---
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, h1=32, h2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Função objetivo para Optuna ---
def objective(trial, X, y, n_epochs=50, val_split=0.2):
    # Espaço de busca
    h1 = trial.suggest_int("h1", 16, 128)
    h2 = trial.suggest_int("h2", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    device = get_device()
    input_dim = X.shape[1]
    output_dim = 1  # ajuste para classificação se necessário

    # Criar modelo
    model = MLP(input_dim=input_dim, output_dim=output_dim, h1=h1, h2=h2, dropout=dropout).to(device)
    criterion = nn.MSELoss()

    # Seleção do otimizador
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # Preparar dataset
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Treinamento
    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    # Avaliação na validação
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            val_loss += criterion(preds, batch_y).item() * batch_X.size(0)
    val_loss /= len(val_loader.dataset)

    return val_loss

# --- Função para otimização ---
def optimize_hyperparams(X, y, n_trials=20, n_epochs=50):
    logger.info("Iniciando otimização de hiperparâmetros com Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, n_epochs=n_epochs), n_trials=n_trials)
    best_params = study.best_trial.params
    logger.info(f"✅ Melhores hiperparâmetros encontrados: {best_params}")
    return best_params
