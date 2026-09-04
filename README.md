# Modelo-de-Classificacao-para-Analise-de-Sentimento

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Pandas](https://img.shields.io/badge/Pandas-2.3.2-purple) ![NumPy](https://img.shields.io/badge/NumPy-2.3.1-orange) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.5-red) ![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-green) ![ScikitiLearn](https://img.shields.io/badge/ScikitiLearn-6.3.0-yellow)

Este projeto tem como propósito desenvolver um modelo de classificação automática de sentimentos  em  textos,  aplicando  conceitos  fundamentais  de  Python,  Machine  Learning  e MLOps. 

O trabalho é estruturado de forma modular, cobrindo todas as etapas do ciclo de vida de um  projeto  de  Ciência  de  Dados.  Após  a  definição  do  problema  e  objetivos,  o  script  realiza  a importação   das   bibliotecas   essenciais   (Pandas,   Scikit-learn,   Seaborn,   entreoutras)   e   o carregamento dos dados a partir de um arquivo CSV contendo reviews de produtos. Em seguida, conduz   uma   análise   exploratória   detalhada   (EDA)   para   compreender   a   distribuição   dos sentimentos e verificar a presença de valores ausentes.

--- 
# Demonstração

O projeto realiza a classificação de sentimentos em avaliações escritas em português, identificando se o conteúdo informado apresenta sentimento positivo ou negativo.

O fluxo principal do projeto é:

Carregamento dos dados > Análise exploratória > Limpeza dos textos > Preparação dos dados > Treinamento dos modelos > Avaliação dos resultados > Classificação de novos textos

O dataset reduzido possui 500 registros. Após a remoção de 12 avaliações sem texto, são utilizadas 488 avaliações válidas, sendo 257 positivas e 231 negativas.

Durante a execução, são gerados gráficos para visualizar a distribuição das classes e o desempenho do modelo.
---

## Distribuição dos sentimentos

![Distribuição dos sentimentos](image/Distribuição%20dos%20sentimentos.png)

---

##  Matriz de Confusão

![Matriz de Confusão](image/Matriz%20de%20Confusão.png)

---

# Tecnologias Utilizadas

- Python 3.11+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

# Estrutura do Projeto

```text
Modelo-de-Classificacao-para-Analise-de-Sentimento/
│
├── data/
│   ├── dataset_ampliado.csv
│   └── dataset.csv
│    
├── image/
│   ├── Distribuição dos sentimentos.png
│   └── Matriz de Confusão.png
│ 
├── models/
│   └── modelo_sentimento_dsa_v1.joblib
│ 
├── src/
│   ├── __init__.py
│   ├── clean_data.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── EDA.py
│   ├── main.py
│   ├── model_evaluation.py
│   ├── model_persistence.py
│   ├── model_training.py
│   └── predicting_sentiments.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Como Executar
## 1. Clone o repositório

```bash
git clone https://github.com/JoaoVitorDC25/Modelo-de-Classificacao-para-Analise-de_Sentimento
```

## 2. Acesse a pasta do projeto

```bash
cd Modelo-de-Classificacao-para-Analise-de-Sentimento
```

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## 4. Execute a aplicação

```bash
python src/main.py
```

---

# Dependências

As dependências do projeto estão listadas no arquivo `requirements.txt`.

Principais bibliotecas utilizadas:

- pandas==2.3.2
- numpy==2.3.1
- matplotlib==3.10.5
- seaborn==0.13.2
- scikit-learn==1.7.2
- joblib

---

# Conceitos Aplicados

Durante o desenvolvimento do projeto foram utilizados conceitos importantes de Análise de Dados, Processamento de Linguagem Natural e Machine Learning:

- *Python:* organização, modularização e desenvolvimento da aplicação.
- *Pandas:* carregamento, inspeção, limpeza e preparação do conjunto de dados.
- *NumPy:* manipulação das avaliações informadas durante a classificação interativa.
- *Matplotlib e Seaborn:* criação dos gráficos de distribuição dos sentimentos e da matriz de confusão.
- *Análise Exploratória de Dados (EDA):* compreensão da estrutura do dataset, distribuição das classes e identificação de valores ausentes.
- *Limpeza de textos:* remoção de acentos, conversão para letras minúsculas, remoção de pontuação e tratamento de espaços excedentes.
- *Processamento de Linguagem Natural (NLP):* preparação dos textos para utilização pelos algoritmos de Machine Learning.
- *TF-IDF:* transformação dos textos em representações numéricas com base na importância das palavras.
- *Pipeline:* integração das etapas de vetorização, padronização e classificação em um único fluxo de processamento.
- *Divisão estratificada:* separação dos dados em treino e teste, preservando a proporção entre avaliações positivas e negativas.
- *GridSearchCV:* comparação de hiperparâmetros com validação cruzada para selecionar a melhor configuração.
- *Regressão Logística, KNN, SVC e Árvore de Decisão:* algoritmos avaliados durante a seleção do modelo.
- *Métricas de classificação:* utilização de acurácia, precisão, recall, F1-score e matriz de confusão para avaliar o desempenho.
- *Persistência do modelo:* utilização do Joblib para salvar e carregar o pipeline treinado.

O projeto busca demonstrar de forma prática como técnicas de Processamento de Linguagem Natural e Machine Learning podem ser aplicadas à classificação de sentimentos em textos.
---

# Autor

**João Vitor Dias**
Técnico em Eletrônica • Estudante de Análise e Desenvolvimento de Sistemas

GitHub: https://github.com/JoaoVitorDC25

Linkedin: <https://www.linkedin.com/in/jo%C3%A3o-vitor-dias-14178a190/?skipRedirect=true>

### Áreas de interesse

-  Ciência de Dados
-  Inteligência Artificial
-  Visão Computacional
-  Desenvolvimento em Python

---

## Projeto em desenvolvimento

Este projeto integra meu portfólio de estudos em Python e Machine Learning.

