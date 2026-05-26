# Naive Bayes - Classificação de Espécies de Iris
# Base de dados: Iris
# Disciplina: Inteligência Artificial

# Leitura da base de dados
base = read.csv('bases/iris/bezdekIris.data', header = FALSE)
colnames(base) = c('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species')

# Verificação dos dados
cat('Dimensões da base:', nrow(base), 'linhas x', ncol(base), 'colunas\n')
cat('Valores ausentes:', sum(is.na(base)), '\n')
cat('\nDistribuição de classes:\n')
print(table(base$species))
cat('\nEstatísticas descritivas:\n')
print(summary(base))

# Encode da variável alvo como fator
base$species = factor(base$species)

# Divisão entre treinamento e teste
library(caTools)
set.seed(1)
divisao = sample.split(base$species, SplitRatio = 0.70)
base_treinamento = subset(base, divisao == TRUE)
base_teste       = subset(base, divisao == FALSE)

cat('\nAmostras de treinamento:', nrow(base_treinamento), '\n')
cat('Amostras de teste:       ', nrow(base_teste), '\n')

# Treinamento do classificador Naive Bayes
library(e1071)
tempo = system.time(
  classificador <- naiveBayes(species ~ ., data = base_treinamento)
)
cat('\nTempo de treinamento:\n')
print(tempo)

# Visualiza as probabilidades aprendidas pelo modelo
cat('\n--- Modelo Naive Bayes ---\n')
print(classificador)

# Previsão no conjunto de teste
previsoes = predict(classificador, newdata = base_teste)

# Matriz de confusão
cat('\n--- Matriz de Confusão ---\n')
matriz_confusao = table(base_teste$species, previsoes)
print(matriz_confusao)

# Métricas de avaliação
library(caret)
cat('\n--- Métricas de Avaliação ---\n')
resultado = confusionMatrix(matriz_confusao)
print(resultado)

# F1-score por classe
precision = resultado$byClass[, 'Pos Pred Value']
recall    = resultado$byClass[, 'Sensitivity']
f1        = 2 * (precision * recall) / (precision + recall)
cat('\nF1-score por classe:\n')
print(round(f1, 4))
cat('\nF1-score medio:', round(mean(f1, na.rm = TRUE), 4), '\n')