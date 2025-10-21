import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from rna_workflow_engine.config import PROCESSED_DATA_DIR

app = typer.Typer()

# ---- Dataset loaders ----


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


DATASET_MAP = {
    "iris": _load_iris,
    "diabetes": _load_diabetes,
    "titanic": _load_titanic,
    "wine": _load_wine,
}


def fetch_dataset(name: str):
    name = name.lower()
    if name not in DATASET_MAP:
        raise ValueError(f"Dataset '{name}' não suportado.")
    X, y = DATASET_MAP[name]()
    logger.info(f"Dataset '{name}' carregado com sucesso")
    return X, y


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
