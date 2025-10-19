import optuna
from sklearn.model_selection import cross_val_score

def optimize_model(model_class, X, y, task, n_trials=50, cv=3):
    def objective(trial):
        params = {}
        if task == "regression":
            if model_class.__name__ == "RandomForestRegressor":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                }
        elif task == "classification":
            if model_class.__name__ == "RandomForestClassifier":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                }
        model = model_class(**params)
        score = cross_val_score(
            model, X, y, cv=cv,
            scoring="accuracy" if task=="classification" else "r2"
        ).mean()
        return score if task=="classification" else -score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
