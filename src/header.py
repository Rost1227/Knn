import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def carregar_dados(caminho_arquivo):
    return pd.read_csv(caminho_arquivo, sep='\t')

def preparar_dados(df):
    X = df.drop(columns=['ID', 'class'])
    y = df['class']
    return X, y

def treinar_modelo(X_train, y_train, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    return knn

def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def mostrar_vizinhos(modelo: KNeighborsClassifier , X_test):
    distances, neighbors = modelo.kneighbors(X_test)
    for idx_teste, (distancias, ids_vizinhos) in enumerate(zip(distances, neighbors)):
        print(f"Instância N{idx_teste + 1}:")
        for distancia, id_vizinho in zip(distancias, ids_vizinhos):
            print(f"  Vizinho ID: {id_vizinho+1}, distância: {distancia:.3f}")
        print("")

def mostrar_acuracias(ks, accuracies):
    print("Acurácias:")
    for k, acc in zip(ks, accuracies):
        print(f"k={k} = {acc:.2f}")
    print("")

