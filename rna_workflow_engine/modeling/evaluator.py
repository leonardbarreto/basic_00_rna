import pandas as pd
import torch
from loguru import logger

from rna_workflow_engine.dataset import fetch_dataset
from rna_workflow_engine.get_model import get_model


def evaluate_model(model_path: str, dataset_name: str):
    X, y = fetch_dataset(dataset_name)
    input_dim = X.shape[1]
    output_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_cls = get_model('mlp', task='regression')
    model = model_cls(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    mse = ((preds.flatten() - y.values) ** 2).mean()
    logger.info(f"Avaliação MSE: {mse:.4f}")
    return {"mse": mse}
