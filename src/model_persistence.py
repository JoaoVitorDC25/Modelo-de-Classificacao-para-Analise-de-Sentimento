import joblib


def save_model(model, model_path: str) -> None:
    """Salva o modelo treinado no caminho informado."""
    joblib.dump(model, model_path)


def load_model(model_path: str):
    """Carrega um modelo salvo em disco."""
    return joblib.load(model_path)
