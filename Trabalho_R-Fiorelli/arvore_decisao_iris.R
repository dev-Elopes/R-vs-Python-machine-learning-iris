# Árvore de Decisão - Classificação de Espécies de Iris
# Base de dados: Iris (bezdekIris.data - versão corrigida)
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

# Treinamento da Árvore de Decisão
# cp = 0: sem poda | maxdepth = 3: profundidade igual ao Python | minsplit = 2: equivalente ao min_samples_split=2
library(rpart)
tempo = system.time(
  classificador <- rpart(formula = species ~ ., data = base_treinamento, method = 'class',
                         control = rpart.control(cp = 0, maxdepth = 3, minsplit = 2))
)
cat('\nTempo de treinamento:\n')
print(tempo)

cat('\n--- Estrutura da Árvore ---\n')
print(classificador)

# Visualização da árvore
library(rpart.plot)
rpart.plot(classificador, type = 2, extra = 104, fallen.leaves = TRUE,
           main = 'Árvore de Decisão - Classificação de Espécies de Iris')

# Previsão no conjunto de teste
previsoes = predict(classificador, newdata = base_teste[-5], type = 'class')

# Matriz de confusão
cat('\n--- Matriz de Confusão ---\n')
matriz_confusao = table(base_teste$species, previsoes)
print(matriz_confusao)

# Métricas de avaliação
library(lattice)
library(ggplot2)
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
