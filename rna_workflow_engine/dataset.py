import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rna_workflow_engine.config import PROCESSED_DATA_DIR

app = typer.Typer()

# ---- Dataset loaders ----


def fetch_dataset(dataset_name: str):
    """
    Carrega dataset padrão do scikit-learn ou personalizado.
    Retorna X, y.
    """
    from sklearn import datasets

    if dataset_name.lower() == "iris":
        data = datasets.load_iris(as_frame=True)
    elif dataset_name.lower() == "wine":
        data = datasets.load_wine(as_frame=True)
    elif dataset_name.lower() == "diabetes":
        data = datasets.load_diabetes(as_frame=True)
    elif dataset_name.lower() == "titanic":
        # Exemplo de dataset customizado
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        data = pd.read_csv(url)
        data = data.dropna(subset=["Age", "Fare", "Pclass", "Sex", "Survived"])
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
        X = data[["Age", "Fare", "Pclass", "Sex"]]
        y = data["Survived"]
        return X, y
    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")

    X = data.data
    y = data.target
    return X, y


def load_dataset_split(dataset_name: str, test_size: float = 0.2, random_state: int = 42):
    """
    Carrega o dataset e retorna o split treino/val, 
    com verificação de classes raras e informações dimensionais.

    Retorna:
        X_train, X_val, y_train, y_val, input_dim, output_dim
    """
    X, y = fetch_dataset(dataset_name)

    # Verifica se é classificação ou regressão (com base no número de classes únicas)
    output_dim = len(pd.Series(y).unique())
    is_classification = output_dim > 2

    # Verifica se pode usar estratificação
    if is_classification:
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            logger.warning(
                "⚠️ Classe com apenas 1 amostra detectada — split sem estratificação.")
            stratify_param = None
        else:
            stratify_param = y
    else:
        stratify_param = None

    # Split seguro
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=stratify_param, random_state=random_state
    )

    input_dim = X.shape[1]
    output_dim = 1 if not is_classification else output_dim

    logger.info(
        f"Dataset '{dataset_name}' carregado com sucesso — "
        f"Treino: {X_train.shape}, Validação: {X_val.shape}, "
        f"Input dim: {input_dim}, Output dim: {output_dim}"
    )

    return X_train, X_val, y_train, y_val, input_dim, output_dim


def _load_iris():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    return data.data, data.target


def _load_diabetes():
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    return data.data, data.target


def _load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    y = df["Survived"]
    return X, y


def _load_wine():
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.data, data.target


def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    output_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = pd.concat([X, y], axis=1)
    df_to_save.to_csv(output_path, index=False)
    logger.info(f"Dataset '{dataset_name}' salvo em {output_path}")
    return output_path


@app.command()
def main(dataset_name: str = "iris"):
    logger.info(f"Iniciando processamento do dataset '{dataset_name}'")
    X, y = fetch_dataset(dataset_name)

    for i in tqdm(range(10)):
        if i == 5:
            logger.info("Etapa intermediária concluída...")

    save_dataset(X, y, dataset_name)
    logger.success(f"Dataset '{dataset_name}' processado e salvo com sucesso.")


if __name__ == "__main__":
    app()
