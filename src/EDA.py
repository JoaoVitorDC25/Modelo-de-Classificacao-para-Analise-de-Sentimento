import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from config import INFO_EDA, GRAPH_EDA

def eda(df_data: pd.DataFrame)->None:
    
    # Informações do dataframe
    if INFO_EDA:
        print(f"\n ----- Dimensões do dataframe: ----- \n\n {df_data.shape[0]} Linhas, {df_data.shape[1]} Colunas")
        print(f"\n ----- Primeiras 5 linhas: ----- \n\n {df_data.head()}")
        print(f"\n ----- Últimas 5 linhas: ----- \n\n {df_data.tail()}")
        print(f"\n ----- Informações do dataframe: ----- \n")
        df_data.info()
        print(f"\n ----- Quantidade de valores ausentes: ----- \n\n {df_data.isnull().sum()}")

    #Graficos do dataframe
    if GRAPH_EDA:
        sn.countplot(x='sentimento', data=df_data)
        plt.title("\nDistribuição dos sentimentos")
        plt.show()