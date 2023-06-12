import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

# Leemos los datos de entrenamiento y de prueba
train_data = pd.read_csv('test.csv')
test_data = pd.read_csv('accident.csv')

# Dividimos los datos en características (X) y objetivo (y)
X = train_data[['speed', 'stop', 'lane', 'accident']]
y = train_data['guilty']

# Dividimos los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo
clf = RandomForestClassifier()

# Definimos los parámetros a probar en la búsqueda en cuadrícula
parameters = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

# Creamos el objeto de búsqueda en cuadrícula
grid_search = GridSearchCV(clf, parameters, cv=5)

# Entrenamos el modelo con la búsqueda en cuadrícula
grid_search.fit(X_train, y_train)

# Imprimimos los mejores parámetros encontrados por la búsqueda en cuadrícula
print("Los mejores parámetros encontrados son: ", grid_search.best_params_)

# Usamos el mejor modelo encontrado para hacer predicciones sobre el conjunto de validación
predictions_val = grid_search.predict(X_val)

# Calculamos e imprimimos el accuracy en el conjunto de validación
accuracy_val = accuracy_score(y_val, predictions_val) * 100
print(f"Accuracy del modelo en el conjunto de validación: {accuracy_val:.2f}%")

# Hacemos predicciones sobre los datos del juzgado
X_test = test_data[['speed', 'stop', 'lane', 'accident']]
predictions_test = grid_search.predict(X_test)

# Imprimimos cada dato del juzgado y su predicción
print(f"\n{'Velocidad (km/h)':<20} {'Se detuvo en señal de stop':<30} {'Usó el carril correcto':<25} {'Hubo un accidente':<20} {'Predicción':<10}")
for test_data, prediction in zip(X_test.values, predictions_test):
    speed, stop, lane, accident = test_data
    pred_str = 'No culpable' if prediction == 0 else 'Culpable' if prediction == 1 else 'Requiere juez'
    print(f"{speed:<20.2f} {stop:<30} {lane:<25} {accident:<20} {pred_str}")

# Calculamos e imprimimos diversas métricas estadísticas
culpable_percentage = (predictions_test == 1).mean() * 100
print(f"\nPorcentaje de casos predichos como culpables: {culpable_percentage:.2f}%")

no_culpable_percentage = (predictions_test == 0).mean() * 100
print(f"Porcentaje de casos predichos como no culpables: {no_culpable_percentage:.2f}%")

juez_required_percentage = (predictions_test == 2).mean() * 100
print(f"Porcentaje de casos que requieren un juez: {juez_required_percentage:.2f}%")

# Imprimimos la lista de todos los casos que requieren un juez
print("\nLista de todos los casos que requieren un juez:")
juez_required_cases = X_test.values[predictions_test == 2]
print(juez_required_cases)