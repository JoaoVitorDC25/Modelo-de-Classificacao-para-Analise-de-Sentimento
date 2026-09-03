import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from config import DATA, MODEL_PATH
from EDA import eda

from predicting_sentiments import predictSent
from data_preparation import prepareData
from model_training import buildGridSearch, trainBestModel
from model_evaluation import evaluate_model
from model_persistence import save_model 

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
    bestModel = trainBestModel(gridSearch, X_Train, y_Train)

    # ----- Model Evaluation -----
    evaluate_model(bestModel, X_Test, y_Test)

    # ----- Save Model -----
    # Se estivermos satisfeitos com a performance do modelo, salvamos em disco
    save_model(bestModel, MODEL_PATH)

    previsoes_treino = bestModel.predict(X_Train)
    previsoes_teste = bestModel.predict(X_Test)

    print(
        f"Acurácia no treino: "
        f"{accuracy_score(y_Train, previsoes_treino):.2%}"
    )

    print(
        f"Acurácia no teste aleatório: "
        f"{accuracy_score(y_Test, previsoes_teste):.2%}"
    )

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_Test, previsoes_teste))

    # Pode deletar o modelo treinado e removê-lo da memória
    del bestModel

    print("Digite um sentimento para classificar (ou 'sair' para encerrar):")
    news_reviews = [input()]
    while news_reviews[-1].strip().lower() != 'sair':
        predictSent([news_reviews[-1]])
        print("Digite um sentimento para classificar (ou 'sair' para encerrar):")
        news_reviews = np.append(news_reviews, input())

    print("Classificação de sentimentos encerrada.")
    
    # Criar novos dados para simular o uso em produção    
    # news_reviews = [
    # "A bateria do celular não dura nada, péssima compra.",
    # "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    # "O serviço de atendimento foi rápido e eficiente.",
    # "Não recomendo, veio faltando peças e a cor estava errada.",
    # "Eu amo a minha namorada.",
    # "Eu não gosto da minha namorada.",
    # "Recebi o monitor hoje E esto muito feliz.",
    # "O produto é bonito e chegou rápido, mas parou de funcionar no segundo dia.",
    # "Produto excelente, funcionou perfeitamente desde o primeiro uso."]
    
    #print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
    #predictSent(news_reviews)
        
if __name__ == "__main__":
    main()