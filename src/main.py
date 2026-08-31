import numpy as np
import pandas as pd
import unicodedata
import seaborn as sn
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATA
from EDA import eda
from clean_data import removeNull, cleanText

from predicting_sentiments import predictSent
from data_preparation import prepareData
from model_training import buildPipeline, getHyperparameterGrid, buildGridSearch, trainBestModel

def main():
    
    df_data=pd.read_csv(DATA)
    
    # ----- EDA -----
    eda(df_data)
    
    # ----- Prepare Data -----
    #X: features
    #y: target
    X, y = prepareData(df_data)
    
    # ----- Split Data -----
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)
    print("Divisão de treino e teste concluida")
    
    # ----- Model Training -----
    gridSearch = buildGridSearch()
    melhor_modelo_dsa = trainBestModel(gridSearch, X_Train, y_Train)
    
    # Previsões no conjunto de teste
    y_pred = melhor_modelo_dsa.predict(X_Test)
    
    # Calcular as métricas de avaliação
    acuracia = accuracy_score(y_Test, y_pred)
    report = classification_report(y_Test, y_pred, target_names = ['Negativo', 'Positivo'])
    
    print(f"\nAcurácia do Modelo: {acuracia:.2%}\n")
    print("Relatório de Classificação:\n")
    print(report)
    
    # Visualizar a Matriz de Confusão
    cm = confusion_matrix(y_Test, y_pred)
    sn.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = ['Negativo', 'Positivo'],
                yticklabels = ['Negativo', 'Positivo'])
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()
    
    # Se estivermos satisfeitos com a performance do modelo, salvamos em disco
    joblib.dump(melhor_modelo_dsa, 'modelo_sentimento_dsa_v1.joblib')
    # Pode deletar o modelo treinado e removê-lo da memória
    del melhor_modelo_dsa
    
    # Carregar o modelo a partir do disco
    modelo_dsa_deploy = joblib.load('modelo_sentimento_dsa_v1.joblib')
    # Criar novos dados para simular o uso em produção
    novos_reviews = [
    "A bateria do celular não dura nada, péssima compra.",
    "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    "O serviço de atendimento foi rápido e eficiente.",
    "Não recomendo, veio faltando peças e a cor estava errada.",
    "Eu amo a minha namorada.",
    "Eu não gosto da minha namorada.",
    "Recebi o monitor hoje E esto muito feliz."]
    
    print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
    predictSent(novos_reviews)
        
if __name__ == "__main__":
    main()

