from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def buildPipeline() -> Pipeline:
     """Cria o pipeline com as mesmas etapas e parâmetros do código original."""
     return Pipeline([
         ('tfidf', TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
         ('scaler', StandardScaler(with_mean=False)),
         ('classifier',LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
         ])

def getHyperparameterGrid() -> list[dict]:
    return [
        {
            'classifier': [
                LogisticRegression(
                    solver='liblinear',
                    random_state=42
                )
            ],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__max_iter': [5000, 6000],
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        },
        {
            'classifier': [
                KNeighborsClassifier()
            ],
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        },
        {
            'classifier': [
                SVC()
            ],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear'],
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        },
        {
            'classifier': [
                DecisionTreeClassifier(
                    random_state=42
                )
            ],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }
    ]

def buildGridSearch() -> GridSearchCV:
    """Configura o GridSearchCV preservando a configuração original."""
    pipeline = buildPipeline()
    parametros_grid = getHyperparameterGrid()

    return GridSearchCV(
        pipeline,               # Pipeline com as etapas de pré-processamento e modelo
        parametros_grid,        # Dicionário com as combinações de hiperparâmetros a serem testadas
        cv=5,                   # Número de divisões para validação cruzada (5-fold cross-validation)
        n_jobs=-1,              # Usa todos os núcleos disponíveis do processador para acelerar o processo
        scoring='recall',     # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
        verbose=1               # Nível de detalhamento do output durante a execução (1 exibe progresso básico)
    )

def trainBestModel(gridSearch, X_Train, y_Train):
    """Treina o grid e devolve o melhor estimador encontrado."""
    print("\nIniciando o treinamento do modelo com otimização de hiperparâmetros...\n")
    gridSearch.fit(X_Train, y_Train)

    print("\nMelhores hiperparâmetros encontrados:\n")
    print(gridSearch.best_params_)

    bestModel = gridSearch.best_estimator_
    print("\nMelhor modelo treinado com sucesso!\n")
    print("Resumo do Melhor Modelo: \n")
    print(bestModel.named_steps['classifier'])
    
    return gridSearch.best_estimator_