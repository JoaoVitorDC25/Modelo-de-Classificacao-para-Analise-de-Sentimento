import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import GRAPH_EVALUATE, PATH_GRAPH

def evaluate_model(bestModel, X_Test, y_Test) -> None:
    """Avalia o modelo e exibe as mesmas métricas e gráfico do fluxo original."""
    # Previsões no conjunto de teste
    y_pred = bestModel.predict(X_Test)

    # Calcular as métricas de avaliação
    acuracia = accuracy_score(y_Test, y_pred)
    report = classification_report(
        y_Test,
        y_pred,
        target_names=['Negativo', 'Positivo']
    )

    print(f"\nAcurácia do Modelo: {acuracia:.2%}\n")
    print("Relatório de Classificação:\n")
    print(report)

    if GRAPH_EVALUATE:
        # Visualizar a Matriz de Confusão
        cm = confusion_matrix(y_Test, y_pred)
        sn.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo']
        )
        plt.xlabel('Previsão')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão')
        
        plt.tight_layout()
        plt.savefig(PATH_GRAPH / f"Matriz de Confusão.png", dpi=300, bbox_inches='tight')
        plt.show()