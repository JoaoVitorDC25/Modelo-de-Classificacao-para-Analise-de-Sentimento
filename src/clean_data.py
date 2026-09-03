import pandas as pd
import unicodedata
import re


def removeNull(df_data: pd.DataFrame)-> pd.DataFrame:
    """
    Função para remover textos nulos
    """
    
    print(f"\n ----- Tamanho original do DF: ----- \n\n {len(df_data)}")
    df_data.dropna(subset = ['texto_review'], inplace=True)
    print(f"\n ----- Tamanho após remover nulos: ----- \n\n {len(df_data)}")
    
    return df_data
    
def cleanText(text):
    """
    Função completa de limpeza do texto:
    1. Remove acentos
    2. Converte minúsculas
    3. Remove pontuação
    4. Remove espaços extras
    """
    
    if not isinstance (text, str):
        return ""
    
    #1. Remove acentos
    """
        Normalization Form KD (Forma de Normalização KD ou Decomposição de Compatibilidade), 
        um padrão do Unicode usado para transformar textos em uma representação uniforme
    """
    text_without_accents = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) !='Mn' )
    
    #2. Converte minúsculas
    text_lower = text_without_accents.lower()
    
    #3. Remove pontuação
    text_without_dots = re.sub(r'[^a-z\s]', ' ',text_lower)
    
    #4. Remove espaços extras
    text_clean = re.sub(r'\s+', ' ', text_without_dots).strip()
        
    return text_clean