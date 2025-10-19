from sklearn.model_selection import train_test_split, cross_val_score
from loguru import logger
from ai_workflow_engine.experiments.mlflow_utils import log_params, log_metrics, log_model

def train_model(model, X, y, task: str, test_size=0.2, cv=5, stratify=None, mlflow_run=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    scoring = "accuracy" if task.lower() == "classification" else "r2"
    logger.info("Executando cross-validation com {} folds...", cv)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)

    logger.info("Treinando modelo no conjunto completo...")
    model.fit(X_train, y_train)

    if mlflow_run:
        log_params(model.get_params())
        log_metrics({
            f"cv_mean_{scoring}": cv_scores.mean(),
            f"cv_std_{scoring}": cv_scores.std()
        })
        log_model(model)

    return model, cv_scores, X_test, y_test
