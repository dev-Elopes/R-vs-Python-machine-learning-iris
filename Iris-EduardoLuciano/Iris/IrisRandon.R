#carregando os pacotes
library(caret)
library(randomForest)

#Lendo os dados ja previamente separados e tratados
dados_treino <- read.csv("iris_treino.csv", stringsAsFactors = TRUE)
dados_teste <- read.csv("iris_teste.csv", stringsAsFactors = TRUE)

# Configurando a Validação Cruzada
controle <- trainControl(method = "cv", number = 10)

#Medindo o tempo de execução
tempo_execucao_rf <- system.time({
  #treinando o modelo
  modelo_rf <- train(
    class ~ ., 
    data = dados_treino, 
    method = "rf",
    trControl = controle
  )
})

print(tempo_execucao_rf) # Mostra o tempo de Execução
print(modelo_rf) # Mostra o melhor modelo encontrado

# Fazendo as previsões nos dados de teste
previsoes_rf <- predict(modelo_rf, newdata = dados_teste)

# Mostra as metricas de cada flor
resultado <- confusionMatrix(previsoes_rf, dados_teste$class, mode = "everything")
print(resultado)
