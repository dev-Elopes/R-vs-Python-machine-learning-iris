#carregando os pacotes
library(caret)
library(nnet)

#Lendo os dados ja previamente separados e tratados
dados_treino <- read.csv("iris_treino.csv", stringsAsFactors = TRUE)
dados_teste <- read.csv("iris_teste.csv", stringsAsFactors = TRUE)

# Configurando a Validação Cruzada
controle <- trainControl(method = "cv", number = 10)

#Medindo o tempo de execução
tempo_execucao_rf <- system.time({
  #treinando o modelo
  modelo_nn <- train(
    class ~ ., 
    data = dados_treino, 
    method = "nnet", 
    preProcess = c("center", "scale"), # Centraliza (média 0) e padroniza (desvio 1) os dados
    trace = FALSE,                     # Evita que o console fique cheio de textos durante o treino
    trControl = controle
  )
})

print(tempo_execucao_nn) # Mostra o tempo de Execução
print(modelo_nn) # Mostra o melhor modelo encontrado

# Fazendo as previsões nos dados de teste
previsoes_nn <- predict(modelo_nn, newdata = dados_teste)

# Mostra as metricas de cada flor
resultado <- confusionMatrix(previsoes_nn, dados_teste$class, mode = "everything")
print(resultado)
