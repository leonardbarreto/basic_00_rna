# --- utils.py ---
import torch

def get_device():
    """
    Retorna o dispositivo disponível: 'cuda' se GPU estiver disponível, caso contrário 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
