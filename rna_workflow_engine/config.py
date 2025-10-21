from pathlib import Path
from loguru import logger
import os

# --- Diretório raiz do projeto ---
PROJ_ROOT = Path(__file__).resolve().parent.parent
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# --- Diretórios de dados ---
RAW_DATA_DIR = PROJ_ROOT / "data/raw"
INTERIM_DATA_DIR = PROJ_ROOT / "data/interim"
PROCESSED_DATA_DIR = PROJ_ROOT / "data/processed"

# --- Diretórios de modelos e relatórios ---
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# --- Configuração segura do Loguru ---
# Remove todos os handlers existentes para evitar erros
logger.remove()  

# Adiciona handler padrão para console com nível INFO
logger.add(
    sink=os.sys.stdout,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
)
