import pandas as pd
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# ------ FUNÇÃO DE FORMATAÇÃO ------
def mostra_resultados(y_true, y_pred, tempo):
    print("\n--- Matriz de Confusão ---")
    cm = confusion_matrix(y_true, y_pred)
    
    # transpondo a matriz (igual o R faz por padrao)
    cm_t = cm.T
    
    # Montando a matriz na mão usando tabulacao (\t) pra alinhar as colunas no terminal
    print("Previsão \t setosa \t versicolor \t virginica")
    print(f"setosa \t\t {cm_t[0][0]} \t\t {cm_t[0][1]} \t\t {cm_t[0][2]}")
    print(f"versicolor \t {cm_t[1][0]} \t\t {cm_t[1][1]} \t\t {cm_t[1][2]}")
    print(f"virginica \t {cm_t[2][0]} \t\t {cm_t[2][1]} \t\t {cm_t[2][2]}")

    # calculando a acurácia direto pelo sklearn
    acc = accuracy_score(y_true, y_pred)

    # usei o round() pra travar as casas decimais e não poluir a tela com dizimas periódicas
    print(f"\n-> Tempo de treino: {round(tempo, 5)}s")
    print(f"-> Acurácia geral: {round(acc, 4)}")
    
    print("\n--- Métricas por classe ---")
    # Extraindo os dados detalhados
    relatorio = classification_report(y_true, y_pred, output_dict=True)
    
    # Pegando as métricas separadas de cada flor e printando na mao
    # o ':.4f' aqui serve pra cravar em 4 casas decimais e deixar o output padronizado
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for c in classes:
        p = relatorio[c]['precision']
        r = relatorio[c]['recall']
        f1 = relatorio[c]['f1-score']
        print(f"[{c}] Precisão: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        
    print("\nMédias Macro:")
    print(f"Precisão Macro: {relatorio['macro avg']['precision']:.4f}")
    print(f"Recall Macro: {relatorio['macro avg']['recall']:.4f}")
    print(f"F1-Score Macro: {relatorio['macro avg']['f1-score']:.4f}")
    print("-" * 60)


#
#------ INÍCIO DO CÓDIGO ------
# 

# 1. Carregando as bases de dados
df_treino = pd.read_csv('iris_treino.csv')
df_teste = pd.read_csv('iris_teste.csv')

X_train = df_treino.drop(columns=['class'])
y_train = df_treino['class']

X_test = df_teste.drop(columns=['class'])
y_test = df_teste['class']

print("\n>>> RODANDO MODELO 1: NAIVE BAYES")

# Criando e treinando o Naive Bayes medindo o tempo
nb = GaussianNB()
t0 = time.time()
nb.fit(X_train, y_train)
t_nb = time.time() - t0

# Previsão e Resultados
preds_nb = nb.predict(X_test)
mostra_resultados(y_test, preds_nb, t_nb)


print("\n>>> RODANDO MODELO 2: ARVORE DE DECISAO")

# Criando a Árvore de Decisão com "max_depth=3" para evitar overfitting
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
t1 = time.time()
dt.fit(X_train, y_train)
t_dt = time.time() - t1

# Previsão e Resultados
preds_dt = dt.predict(X_test)
mostra_resultados(y_test, preds_dt, t_dt)

# Salvando a imagem da árvore 
plt.figure(figsize=(12, 8))
plot_tree(dt, 
          feature_names=list(X_train.columns), 
          class_names=list(nb.classes_), 
          filled=True,
          #ajustes para deixar alinhado nos conformes
          rounded=True,
          fontsize=10) 

plt.tight_layout() # força o python a organizar as margens pra caber tudo
plt.savefig('arvore_python.png', dpi=300)
print("\nO Gráfico da árvore de decisão foi salvo como 'arvore_python.png' na sua pasta!")