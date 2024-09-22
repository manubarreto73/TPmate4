import pandas as pd
from regresion_lineal import regresion_lineal
from typing import Dict

# Ruta al archivo CSV
file_path = 'content\players_21.csv'

# Leer el archivo CSV
df = pd.read_csv(file_path)

# Imprimimos la suma de los valores de la columna defending_marking, reemplazando aquellos que sean null por 0
print('La suma de todos los valores de defending_making es ' + str(df['defending_marking'].fillna(0).sum()))

# Definir las columnas para las que se van a hacer regresiones
columnas = [
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 
    'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 
    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 
    'movement_agility', 'movement_reactions', 'movement_balance', 
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
    'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 
    'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
    'mentality_composure', 'defending_standing_tackle', 
    'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 
    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'
]

# Definir la columna dependiente (y) para todas las regresiones
columna_y = 'value_eur'

# Lista para guardar los coeficientes de determinación R^2
valores_R2: Dict[str, float] = {}

# Loop por cada columna especificada en 'columnas'
for columna_x in columnas:

    df[columna_x] = pd.to_numeric(df[columna_x], errors='coerce')
    df[columna_y] = pd.to_numeric(df[columna_y], errors='coerce')

    # Extraer los valores de las columnas de interes
    X = df[columna_x].values
    Y = df[columna_y].values

    # Llamar a la función de regresión lineal
    R = regresion_lineal(columna_x, X, Y)
    
    # Guardar el valor de R2 en el diccionario con la columna como clave
    valores_R2[columna_x] = R**2 if R is not None else 0.0 # Me aseguro que R es un float
    
    # Mostrar resultados por columna
    print(f"Regresión lineal para '{columna_x}' vs '{columna_y}':")
    print(f"Coeficiente de determinación (R^2): {R**2:.5f}\n")

# Informar la característica más relevante
columna_max = max(valores_R2, key=lambda k: valores_R2.get(k, 0.0))
print(f'El atributo que más determina el precio es: {columna_max}')
