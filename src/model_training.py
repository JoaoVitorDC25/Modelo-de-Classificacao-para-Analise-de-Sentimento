from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def buildPipeline() -> Pipeline:
    """Cria o pipeline com as mesmas etapas e parâmetros do código original."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
        ('scaler', StandardScaler(with_mean=False)),
        ('logreg',LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
        ])

def getHyperparameterGrid() -> dict:
    """Retorna o grid de hiperparâmetros utilizado na otimização."""
    return {
        'tfidf__max_features': [500, 1000, 2000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'logreg__C': [0.1, 1, 10],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__max_iter': [5000, 6000]
    }

def buildGridSearch() -> GridSearchCV:
    """Configura o GridSearchCV preservando a configuração original."""
    pipeline = buildPipeline()
    parametros_grid = getHyperparameterGrid()

    return GridSearchCV(
        pipeline,               # Pipeline com as etapas de pré-processamento e modelo
        parametros_grid,        # Dicionário com as combinações de hiperparâmetros a serem testadas
        cv=5,                   # Número de divisões para validação cruzada (5-fold cross-validation)
        n_jobs=-1,              # Usa todos os núcleos disponíveis do processador para acelerar o processo
        scoring='accuracy',     # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
        verbose=1               # Nível de detalhamento do output durante a execução (1 exibe progresso básico)
    )

def trainBestModel(gridSearch, X_Train, y_Train):
    """Treina o grid e devolve o melhor estimador encontrado."""
    print("\nIniciando o treinamento do modelo com otimização de hiperparâmetros...\n")
    gridSearch.fit(X_Train, y_Train)

    print("\nMelhores hiperparâmetros encontrados:\n")
    print(gridSearch.best_params_)

    return gridSearch.best_estimator_