import pandas as pd
import numpy as np

# Establecemos la semilla para la reproducibilidad
np.random.seed(0)

# Generamos 1000 muestras aleatorias para cada característica
speed = np.random.uniform(0, 120, 1000) # velocidades entre 0 y 120 km/h
stop = np.random.choice([0, 1], 1000) # si se detuvo o no en una señal de stop
lane = np.random.choice([0, 1], 1000) # si usó el carril correcto o no
accident = np.random.choice([0, 1], 1000) # si hubo un accidente o no

# Definimos las reglas para determinar si una persona es culpable o no
guilty = (speed > 75) | (stop == 0) | (lane == 0)

# Si hubo un accidente pero los otros parámetros indican que no es culpable, se requiere un juez
guilty = np.where((accident == 1) & (guilty == 0), 2, guilty)

# Creamos el DataFrame y guardamos los datos en un CSV
df = pd.DataFrame({'speed': speed, 'stop': stop, 'lane': lane, 'accident': accident, 'guilty': guilty})
df.to_csv('test.csv', index=False)


# Generamos un segundo conjunto de datos para simular el entorno real (juzgado.csv)
n_juzgado = 200

speed_juzgado = np.random.uniform(50, 100, n_juzgado)
stop_juzgado = np.random.choice([0, 1], n_juzgado)
lane_juzgado = np.random.choice([0, 1], n_juzgado)
accident_juzgado = np.random.choice([0, 1], n_juzgado)

# No necesitamos calcular 'guilty' para estos datos, ya que eso es lo que queremos predecir
juzgado_data = pd.DataFrame({'speed': speed_juzgado, 'stop': stop_juzgado, 'lane': lane_juzgado, 'accident': accident_juzgado})

# Guardamos los datos en un archivo CSV
juzgado_data.to_csv('accident.csv', index=False)
