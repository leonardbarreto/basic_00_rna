# rna_workflow_engine/utils.py
import torch

def get_device(verbose: bool = True):
    """
    Retorna o dispositivo de execução disponível para PyTorch.

    Esta função detecta automaticamente se há uma GPU CUDA ou Apple MPS
    disponível. Caso contrário, utiliza CPU.

    Parâmetros
    ----------
    verbose : bool, opcional
        Exibe informações sobre o dispositivo detectado, por padrão True.

    Retorna
    -------
    torch.device
        Objeto representando o dispositivo disponível ('cuda', 'mps' ou 'cpu').

    Referências
    -----------
    - PyTorch Documentation: https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
    - Best practices for GPU usage in deep learning.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("✅ GPU Apple MPS detectada.")
    else:
        device = torch.device("cpu")
        if verbose:
            print("⚙️  Executando em CPU.")

    return device
