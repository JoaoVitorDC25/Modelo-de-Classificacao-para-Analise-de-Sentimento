from pathlib import Path

#/main.py
DATA = '../data/dataset.csv'
MODEL_PATH = '../models/modelo_sentimento_dsa_v1.joblib'

#/EDA.py
#Habilita / desabilita a visualização dos dados no terminal
#INFO_EDA = True
INFO_EDA = False

#Habilita / desabilita a geração de gráficos
GRAPH_EDA = True
#GRAPH_EDA = False

#/model_evaluation.py
GRAPH_EVALUATE = True
#GRAPH_EVALUATE = False
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_GRAPH = PROJECT_ROOT / "image"

PATH_GRAPH.mkdir(parents=True, exist_ok=True)

