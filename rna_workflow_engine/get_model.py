import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, h1=64, h2=64, dropout=0.2):
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

def get_model(model_type: str, input_dim=None, output_dim=None, task="regression"):
    """
    Retorna a classe do modelo solicitada.
    Atualmente só MLP está implementado.
    """
    model_type = model_type.lower()
    if model_type == "mlp":
        if input_dim is None or output_dim is None:
            raise ValueError("Para MLP, é necessário input_dim e output_dim")
        return MLP(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Modelo '{model_type}' não suportado")
