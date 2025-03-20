from header import *

accuraciesO2f = []
accuraciesO11f = []
accuraciesN2f = []
accuraciesN11f = []

ks = [1, 3, 5, 7]

training_o2f = carregar_dados('data/Dados_Originais_2Features/TrainingData_2F_Original.txt')
teste_o2f = carregar_dados('data/Dados_Originais_2Features/TestingData_2F_Original.txt')
X_train, y_train = preparar_dados(training_o2f)
X_test, y_test = preparar_dados(teste_o2f)

print("Dados Originais 2 Features")
for k in ks:
    o2f_k = treinar_modelo(X_train, y_train, k)
    accuracy = avaliar_modelo(o2f_k, X_test, y_test)
    print("-------------------------------------------")
    print(f'Acurácia com k={k}: {accuracy:.2f}')
    mostrar_vizinhos(o2f_k, X_test)
    accuraciesO2f.append(accuracy)
mostrar_acuracias(ks, accuraciesO2f)

training_o11f = carregar_dados('data/Dados_Originais_11Features/TrainingData_11F_Original.txt')
teste_o11f = carregar_dados('data/Dados_Originais_11Features/TestingData_11F_Original.txt')
X_train, y_train = preparar_dados(training_o11f)
X_test, y_test = preparar_dados(teste_o11f)

print("Dados Originais 11 Features")
for k in ks:
    o11f_k = treinar_modelo(X_train, y_train, k)
    accuracy = avaliar_modelo(o11f_k, X_test, y_test)
    print("-------------------------------------------")
    print(f'Acurácia com k={k}: {accuracy:.2f}')
    mostrar_vizinhos(o11f_k, X_test)
    accuraciesO11f.append(accuracy)
mostrar_acuracias(ks, accuraciesO11f)

training_n2f = carregar_dados('data/Dados_Normalizados_2Features/TrainingData_2F_Norm.txt')
teste_n2f = carregar_dados('data/Dados_Normalizados_2Features/TestingData_2F_Norm.txt')
X_train, y_train = preparar_dados(training_n2f)
X_test, y_test = preparar_dados(teste_n2f)

print("Dados Normalizados 2 Features")
for k in ks:
    n2f_k = treinar_modelo(X_train, y_train, k)
    accuracy = avaliar_modelo(n2f_k, X_test, y_test)
    print("-------------------------------------------")
    print(f'Acurácia com k={k}: {accuracy:.2f}')
    mostrar_vizinhos(n2f_k, X_test)
    accuraciesN2f.append(accuracy)
mostrar_acuracias(ks, accuraciesN2f)

training_n11f = carregar_dados('data/Dados_Normalizados_11Features/TrainingData_11F_Norm.txt')
teste_n11f = carregar_dados('data/Dados_Normalizados_11Features/TestingData_11F_Norm.txt')
X_train, y_train = preparar_dados(training_n11f)
X_test, y_test = preparar_dados(teste_n11f)

print("Dados Normalizados 11 Features")
for k in ks:
    n11f_k = treinar_modelo(X_train, y_train, k)
    accuracy = avaliar_modelo(n11f_k, X_test, y_test)
    print("-------------------------------------------")
    print(f'Acurácia com k={k}: {accuracy:.2f}')
    mostrar_vizinhos(n11f_k, X_test)
    accuraciesN11f.append(accuracy)
mostrar_acuracias(ks, accuraciesN11f)

accuracies_dict = {
    'Original 2 features': accuraciesO2f,
    'Original 11 features': accuraciesO11f,
    'Normalizado 2 features': accuraciesN2f,
    'Normalizado 11 features': accuraciesN11f
}

print("\nAcurácias Finais:")
for dataset, accuracies in accuracies_dict.items():
    print(f"{dataset}:")
    for k, acc in zip(ks, accuracies):
        print(f"k={k} = {acc:.2f}")
    print("")