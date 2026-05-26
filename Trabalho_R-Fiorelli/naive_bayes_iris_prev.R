# Naive Bayes - Classificação de Espécies de Iris
# Base de dados: Iris (já separada previamente)
# Disciplina: Inteligência Artificial

# Leitura da base de dados
base_treinamento = read.csv('bases/iris/iris_treino.csv')
base_teste       = read.csv('bases/iris/iris_teste.csv')

cat('\nAmostras de treinamento:', nrow(base_treinamento), '\n')
cat('Amostras de teste:       ', nrow(base_teste), '\n')

# Encode da variável alvo como fator
base_treinamento$class = factor(base_treinamento$class)
base_teste$class       = factor(base_teste$class)

# Treinamento do classificador Naive Bayes
library(e1071)
tempo = system.time(
  classificador <- naiveBayes(class ~ ., data = base_treinamento)
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
matriz_confusao = table(base_teste$class, previsoes)
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
