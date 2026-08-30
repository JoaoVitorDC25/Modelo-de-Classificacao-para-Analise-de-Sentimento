import pandas as pd

from sklearn.model_selection import train_test_split

from clean_data import cleanText, removeNull

def prepareData(df_data: pd.DataFrame):
    """Executa a limpeza dos dados e a criação das variáveis X e y."""
    df_data_Nnull = removeNull(df_data)
    df_data_Nnull['text_clean'] = df_data_Nnull['texto_review'].apply(cleanText)
    print(df_data_Nnull.head())

    # ----- Feature Eng -----
    df_data_Nnull['sentimento_label'] = df_data_Nnull['sentimento'].map({'positivo': 1,'negativo': 0})

    # Definir X (feature/entrada) e y (target/saída)
    X = df_data_Nnull['text_clean']
    y = df_data_Nnull['sentimento_label']

    return X, y