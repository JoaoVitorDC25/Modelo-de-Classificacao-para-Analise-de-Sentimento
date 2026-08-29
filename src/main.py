import numpy as np
import pandas as pd
import unicodedata
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATA

def main():
    df_arquivo=pd.read_csv(DATA)
    
    print(df_arquivo.shape)

if __name__ == "__main__":
    main()

