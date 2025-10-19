from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def get_model(model_type: str, task: str):
    model_type = model_type.lower()
    if task == "regression":
        mapping = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "svr": SVR,
            "knn": KNeighborsRegressor
        }
    else:
        mapping = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svc": SVC,
            "knn": KNeighborsClassifier
        }
    if model_type not in mapping:
        raise ValueError(f"Modelo '{model_type}' não suportado para tarefa '{task}'")
    return mapping[model_type]
