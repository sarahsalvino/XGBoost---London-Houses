# London Houses - Previsao de Preço de Imóveis com XGBoost

Modelo de regressão para prever o preço de imóveis em Londres utilizando XGBoost Regressor,
com OneHotEncoder para tratamento de variáveis categóricas e RandomizedSearchCV para
otimização de hiperparametros.

---

## Objetivo

Estudar o funcionamento do modelo XGBoost e seus parametros aplicados a um problema real
de regressão, e explorar a otimização de hiperparametros com RandomizedSearch como
estudo complementar.

---

## Estrutura do Projeto

```
projeto
  London_Houses.ipynb       Notebook principal
  london_houses.csv         Dataset
  README.md
```

---

## Dataset

- 1.000 registros de imóveis em Londres
- 17 colunas originais
- Sem valores nulos e sem duplicados
- Variavel alvo: Price (£)

### Colunas utilizadas no modelo

| Coluna | Descrição |
|--------|-----------|
| Address | Endereço do imóvel |
| Bedrooms | Número de quartos |
| Bathrooms | Número de banheiros |
| Square Meters | Metragem quadrada |
| Building Age | Idade do edificio (anos) |
| Garden | Possui jardim (Yes/No) |
| Garage | Possui garagem (Yes/No) |
| Property Type | Tipo do imóvel (Apartment, Semi-Detached, Detached House) |
| Balcony | Tipo de sacada |
| Interior Style | Estilo de interiores (Modern, Classic, Industrial, Minimalist) |
| View | Vista do imovel (City, Park, Garden, Sea, Street) |
| Building Status | Estado do edificio (New, Old, Renovated) |

### Colunas não utilizadas

| Coluna | Motivo |
|--------|--------|
| Neighborhood | Não incluida nas features do modelo |
| Floors | Não incluida nas features do modelo |
| Heating Type | Não incluida nas features do modelo |
| Materials | Não incluida nas features do modelo |

---

## Pré-processamento

As variáveis categóricas foram transformadas com OneHotEncoder via ColumnTransformer,
criando colunas binarias para cada categoria e evitando hierarquias falsas entre os valores.

Divisão dos dados: 80% treino (800 amostras) e 20% teste (200 amostras).

---

## Modelos Aplicados

### Etapa 1 - XGBoost Base

Modelo treinado com parâmetros iniciais para estabelecer uma baseline de performance.

```python
XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### Etapa 2 - RandomizedSearchCV

Busca aleatória por melhores hiperparametros com 20 combinacoes testadas e validação
cruzada de 5 folds (100 fits no total).

Melhores parâmetros encontrados:
- n_estimators: 500
- max_depth: 3
- learning_rate: 0.05
- subsample: 1.0
- colsample_bytree: 1.0
- reg_alpha: 0
- reg_lambda: 5

---

## Resultados

| Métrica | Modelo Base | RandomizedSearch |
|---------|------------|-----------------|
| MAPE    | 19.89%     | 19.95%          |
| SMAPE   | 19.27%     | 19.18%          |
| MAE     | £356,372   | £355,061        |
| RMSE    | £462,880   | £460,407        |
| R2      | 0.734      | 0.737           |

### Sobre as métricas

- MAPE: erro percentual médio em relação ao preço real
- SMAPE: versão simétrica do MAPE, mais estável para preços próximos de zero
- MAE: erro médio absoluto em libras esterlinas
- RMSE: penaliza erros grandes mais severamente que o MAE
- R2: proporção da variação nos preços explicada pelo modelo

### Interpretação

O modelo base atingiu R2 de 0.734, explicando 73.4% da variação nos preços. O erro
médio de £356,372 em imóveis entre £500k e £2.5M representa um erro de aproximadamente
19-20%, esperado para um dataset com alta variabilidade de preços em diferentes
bairros de Londres.

O RandomizedSearch entregou melhora marginal em todas as métricas, com o R2 subindo
de 0.734 para 0.737. A proximidade dos resultados indica que o modelo base já estava
bem configurado, e que aumentar n_iter no RandomizedSearch poderia explorar melhor
o espaco de parametros.

---

## Exemplo de Previsão

```python
nova_casa = {
    "Address": "Camden",
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Square Meters": 120,
    "Building Age": 10,
    "Garden": "Yes",
    "Garage": "No",
    "Property Type": "Detached",
    "Balcony": "No",
    "Interior Style": "Modern",
    "View": "City",
    "Building Status": "New"
}

Preço estimado: £1,534,109.50
```

---

## Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- XGBoost
- Scikit-learn

---

## Como Executar

1. Clone o repositorio
```bash
git clone https://github.com/sarahsalvino/XGBoost---London-Houses.git
```

2. Instale as dependências
```bash
pip install pandas numpy matplotlib xgboost scikit-learn
```

3. Abra o notebook
```bash
jupyter notebook London_Houses.ipynb
```

Certifique-se de que o arquivo london_houses.csv esta na mesma pasta do notebook antes de rodar.
