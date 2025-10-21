import matplotlib.pyplot as plt
from loguru import logger

def plot_training_validation(train_losses, val_losses, fold: int):
    """
    Gera gráfico de perdas de treino e validação por fold.
    
    Args:
        train_losses (list[float]): lista de losses de treino por batch
        val_losses (list[float]): lista de losses de validação por batch
        fold (int): número do fold
    Returns:
        fig (matplotlib.figure.Figure): figura gerada
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Treino', color='blue')
    ax.plot(val_losses, label='Validação', color='orange')
    ax.set_title(f'Fold {fold} - Treino vs Validação')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    logger.info(f"Gráfico de Treino/Validação gerado para fold {fold}")
    return fig
