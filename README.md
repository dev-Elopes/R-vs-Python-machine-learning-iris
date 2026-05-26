# Estudo Comparativo: Machine Learning no Iris Dataset (R vs. Python)

## Descrição do Projeto
Este repositório contém o código-fonte e a metodologia utilizados em um estudo comparativo de algoritmos de classificação supervisionada. O objetivo foi avaliar o desempenho, a reprodutibilidade e o trade-off computacional entre **R** e **Python** utilizando o clássico Iris Dataset.

## Algoritmos Implementados
Foram comparados 4 modelos principais em ambas as linguagens:
* **Naive Bayes** (Probabilístico)
* **Árvore de Decisão** (Regras Lógicas)
* **Random Forest** (Ensemble)
* **Rede Neural (MLP)** (Deep Learning Básico)

## Principais Destaques da Metodologia
- **Isonomia:** Divisão 70/30 estratificada com semente (seed) fixada em 42 para garantir que ambos os ambientes processassem as mesmas instâncias de treino/teste.
- **Engenharia de Dados:** Aplicação de normalização (StandardScaler) onde aplicável e ajuste de hiperparâmetros (como `cp` no R e `criterion` no Python).
- **Métricas:** Além da acurácia, utilizamos Matriz de Confusão, Precisão, Recall e F1-Score para uma análise equilibrada.

## Estrutura do Repositório
/data       -> Conjunto de dados original.
/r_scripts  -> Códigos de análise e modelagem em R.
/py_scripts -> Códigos de análise e modelagem em Python.
/results    -> Matrizes de confusão e logs de performance.

## Veredito do Estudo
Embora a Rede Neural tenha apresentado alta robustez, a **Árvore de Decisão** destacou-se pela eficiência, entregando resultados comparáveis com um custo computacional drasticamente menor, sendo a escolha ideal em termos de interpretabilidade e velocidade.
