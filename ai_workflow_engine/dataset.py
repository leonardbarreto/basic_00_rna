from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from ai_workflow_engine.config import PROCESSED_DATA_DIR

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

DATASET_MAP = {
    "iris": _load_iris,
    "diabetes": _load_diabetes,
    "titanic": _load_titanic,
}

def fetch_dataset(name: str):
    name = name.lower()
    if name not in DATASET_MAP:
        raise ValueError(f"Dataset '{name}' não suportado.")
    X, y = DATASET_MAP[name]()
    logger.info(f"Dataset '{name}' carregado com sucesso")
    return X, y

def save_dataset(X: pd.DataFrame, y: pd.Series, output_path: Path):
    df_to_save = pd.concat([X, y], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(output_path, index=False)
    logger.info(f"Dataset salvo em {output_path}")

@app.command()
def main(dataset_name: str = "iris", output_path: Path = PROCESSED_DATA_DIR / "dataset.csv"):
    logger.info(f"Iniciando processamento do dataset '{dataset_name}'")
    X, y = fetch_dataset(dataset_name)
    for i in tqdm(range(10)):
        if i == 5:
            logger.info("Etapa intermediária concluída...")
    save_dataset(X, y, output_path)
    logger.success("Processamento de dataset concluído.")

if __name__ == "__main__":
    app()
