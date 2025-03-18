import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df_treino = pd.read_csv('data/Dados_Originais_2Features/TrainingData_2F_Original.txt', sep='\t')
df_teste = pd.read_csv('data/Dados_Originais_2Features/TestingData_2F_Original.txt', sep='\t')

X_train = df_treino.drop(columns=['ID', 'class'])  # Atributos usados na classificação
y_train = df_treino['class']                      # Classe real das instâncias (0 ou 1)

k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

X_test = df_teste.drop(columns=['ID', 'class'])
y_test = df_teste['class']

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia com k={k}: {accuracy:.2f}')

distances, neighbors = knn.kneighbors(X_test)


for idx_teste, (dist, ids_vizinhos) in enumerate(zip(distances, neighbors)):
    print(f"Instância N{idx_teste + 1}:")
    for distancia, id_vizinho in zip(distances[idx_teste], neighbors[idx_teste]):
        id_vizinho = df_treino.iloc[id_vizinho]['ID']
        print(f"  Vizinho ID: {id_vizinho}, distância: {distancia:.3f}")
    print("\n")
