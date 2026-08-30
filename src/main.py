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

def main():
    
    df_data=pd.read_csv(DATA)
    
    # ----- EDA -----
    eda(df_data)
    
    # ----- Prepare Data -----
    #X: features
    #y: target
    X, y = prepareData(df_data)
    
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)
    print("Divisão de treino e teste concluida")
    
    #Pipeline
    pipeline = Pipeline([
    
    ('tfidf', TfidfVectorizer(stop_words = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
    
    ('scaler', StandardScaler(with_mean = False)),
    
    ('logreg', LogisticRegression(solver = 'liblinear', random_state = 42, max_iter = 1000)) 
    ])
    
    # Definir o grid de hiperparâmetros para otimização
    parametros_grid = {
        'tfidf__max_features': [500, 1000, 2000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'logreg__C': [0.1, 1, 10],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__max_iter': [5000, 6000]
    }
    
    # Configurar o GridSearchCV
    grid_search = GridSearchCV(
    pipeline,              # Pipeline com as etapas de pré-processamento e modelo
    parametros_grid,       # Dicionário com as combinações de hiperparâmetros a serem testadas
    cv = 5,                # Número de divisões para validação cruzada (5-fold cross-validation)
    n_jobs = -1,           # Usa todos os núcleos disponíveis do processador para acelerar o processo
    scoring = 'accuracy',  # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
    verbose = 1            # Nível de detalhamento do output durante a execução (1 exibe progresso básico)
    )
    
    print("\nIniciando o treinamento do modelo com otimização de hiperparâmetros...\n")
    grid_search.fit(X_Train, y_Train)
    
    print("\nMelhores hiperparâmetros encontrados:\n")
    print(grid_search.best_params_)
    
    # Obter o melhor modelo
    melhor_modelo_dsa = grid_search.best_estimator_
    
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

