# pipeline.py
import typer
from loguru import logger

from ai_workflow_engine.config import PROCESSED_DATA_DIR
from ai_workflow_engine.dataset import main as dataset_main
from ai_workflow_engine.modeling.train import \
    train_model  # função de treino principal

app = typer.Typer()

@app.command()
def run_pipeline(
    dataset_name: str = "iris",
    model_type: str = "random_forest",
    task: str = "classification",
    test_size: float = 0.2,
    cv: int = 3,
    n_trials: int = 50,
):
    """
    Executa todo o pipeline: processamento do dataset e treinamento do modelo.
    """
    logger.info("==== INICIANDO PIPELINE ====")

    # 1️⃣ Processar dataset
    logger.info(f"Processando dataset '{dataset_name}'")
    #raw_path = PROCESSED_DATA_DIR.parent / "raw" / f"{dataset_name}.csv"
    processed_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"

    dataset_main(dataset_name='iris',output_path=processed_path)
    # 2️⃣ Treinar modelo
    logger.info(f"Treinando modelo '{model_type}' para tarefa '{task}'")
    model, metrics = train_model(
        dataset_name=dataset_name,
        model_type=model_type,
        task=task,
        test_size=test_size,
        cv=cv,
        n_trials=n_trials
    )

    logger.success("🎉 Pipeline concluído! Parabéns! 🍺🍺")

if __name__ == "__main__":
    app()
