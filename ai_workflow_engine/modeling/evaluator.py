# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# def evaluate_model(model, X_test, y_test, task="classification"):
#     if task == "regression":
#         y_pred = model.predict(X_test)
#         metrics = {
#             "r2": r2_score(y_test, y_pred),
#             "mse": mean_squared_error(y_test, y_pred)
#         }
#     else:
#         y_pred = model.predict(X_test)
#         metrics = {
#             "accuracy": accuracy_score(y_test, y_pred)
#         }
#     return metrics

# ai_workflow_engine/modeling/evaluator.py

from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_model(model, X_test, y_test, task: str = "classification"):
    """
    Avalia o modelo conforme o tipo de tarefa (classificação ou regressão).
    """
    if task == "regression":
        y_pred = model.predict(X_test)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred)
        }
    else:  # classification
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred)
        }

    return metrics
