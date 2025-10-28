import io
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch
from config import MODELS_DIR  # supondo que exista MODELS_DIR
from loguru import logger
from mlflow.models import infer_signature

from rna_workflow_engine.config import MODELS_DIR


def start_run(experiment_name: str = "MLP_Classification", run_name: str = None):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run started: {run.info.run_id}")
    return run


def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)
    logger.info(f"Parâmetros logados: {params}")


def log_metrics(metrics: dict, step: int = None):
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)
    logger.info(f"Métricas logadas: {metrics}")


def log_model(model, model_name: str = "model", X_sample=None, y_sample=None, metadata: dict = None):
    """
    Loga um modelo PyTorch no MLflow com:
    - model signature (entrada/saída inferida automaticamente)
    - input_example (para UI e API)
    - schema JSON detalhado (artefato adicional)

    Parâmetros:
    ------------
    model : torch.nn.Module
        Modelo PyTorch treinado.
    model_name : str
        Nome do artefato dentro do MLflow.
    X_sample, y_sample : DataFrame, ndarray ou tensor
        Amostras de entrada e saída (usadas para inferir signature e schema).
    metadata : dict
        Informações adicionais sobre dataset, métricas, etc.
    """
    if not mlflow.active_run():
        raise RuntimeError(
            "Nenhum run ativo. Use mlflow.start_run() antes de chamar log_model().")

    model.eval()

    # 🔹 Infere assinatura do modelo (para UI e serving)
    signature = None
    if X_sample is not None and y_sample is not None:
        try:
            # Converte para tensor, se necessário
            if isinstance(X_sample, torch.Tensor):
                input_ex = X_sample.cpu().detach().numpy()
            else:
                input_ex = getattr(X_sample, "values", X_sample)

            if isinstance(y_sample, torch.Tensor):
                output_ex = y_sample.cpu().detach().numpy()
            else:
                output_ex = getattr(y_sample, "values", y_sample)

            signature = infer_signature(input_ex, output_ex)
        except Exception as e:
            logger.warning(f"Falha ao inferir assinatura: {e}")

    # 🔹 Define exemplo de entrada (para UI e serving)
    input_example = None
    if X_sample is not None:
        try:
            if isinstance(X_sample, torch.Tensor):
                input_example = X_sample[:1].cpu().detach().numpy()
            else:
                input_example = X_sample.head(1) if hasattr(
                    X_sample, "head") else X_sample[:1]
        except Exception as e:
            logger.warning(f"Falha ao definir input_example: {e}")

    # 🔹 Serializa modelo na memória (não salva localmente)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model_bytes = buffer.getvalue()

    # 🔹 Loga o modelo no MLflow com assinatura e exemplo
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=f"models/{model_name}",
        signature=signature,
        input_example=input_example,
        registered_model_name=None
    )

    # 🔹 Gera schema detalhado para registro adicional
    model_schema = {
        "model_name": model_name,
        "model_type": "pytorch",
        "timestamp": datetime.utcnow().isoformat(),
        "mlflow_run_id": mlflow.active_run().info.run_id,
        "framework": "torch",
        "class_name": model.__class__.__name__,
        "parameters_count": sum(p.numel() for p in model.parameters()),
        "signature_inferred": signature is not None,
        "has_input_example": input_example is not None,
    }

    if metadata:
        model_schema.update(metadata)

    # 🔹 Loga schema como artefato adicional (para consulta manual)
    mlflow.log_text(json.dumps(model_schema, indent=2, ensure_ascii=False),
                    artifact_file=f"models/{model_name}/{model_name}_schema.json")

    logger.info(
        f"✅ Modelo '{model_name}' logado no MLflow com assinatura e input_example.")


def log_figure(fig=None, path=None, name: str = None):
    """
    Loga uma figura (Matplotlib ou arquivo existente) no MLflow.

    Parâmetros
    ----------
    fig : matplotlib.figure.Figure, opcional
        Objeto da figura Matplotlib a ser logado.
    path : str ou Path, opcional
        Caminho para o arquivo de imagem (.png, .jpg) já salvo.
    name : str, opcional
        Nome do artifact no MLflow (caso não informado, usa o nome do arquivo).

    Exemplo de uso
    --------------
    # Se você já salvou o arquivo:
    log_figure(path=FIGURES_DIR / "iris_loss_curve.png")

    # Ou se quiser logar direto de uma figura Matplotlib:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    log_figure(fig=fig, name="simple_plot.png")
    """
    try:
        if path is not None:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Arquivo de figura não encontrado: {path}")
                return
            artifact_name = name or path.name
            mlflow.log_artifact(str(path), artifact_path="figures")
            logger.info(
                f"Figura '{artifact_name}' logada no MLflow a partir de arquivo.")
        elif fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            artifact_name = name or "figure.png"
            mlflow.log_image(buf, artifact_file=f"figures/{artifact_name}")
            buf.close()
            logger.info(
                f"Figura '{artifact_name}' logada no MLflow a partir de objeto Matplotlib.")
        else:
            logger.warning(
                "Nenhuma figura ou caminho informado para log_figure().")
    except Exception as e:
        logger.error(f"Erro ao logar figura no MLflow: {e}")


def end_run():
    mlflow.end_run()
    logger.info("MLflow run finalizado")
