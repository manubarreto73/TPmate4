import pandas as pd
from regresion_lineal import regresion_lineal

# Ruta al archivo CSV
file_path = 'players_21.csv'

# Leer el archivo CSV
df = pd.read_csv(file_path)

# Definir las columnas para las que se van a hacer regresiones
columnas = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passin', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strenght', 'power_long_shots', 'mentality_aggresion', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

# Definir la columna dependiente (y) para todas las regresiones
columna_y = 'value_eur'

# Lista para guardar los coeficientes de determinación R^2
valores_R2 = {}

# Loop por cada columna especificada en 'columnas'
for columna_x in columnas:
    try:
        # Extraer los valores de las columnas de interes
        X = df[columna_x].values
        y = df[columna_y].values

        # Llamar a la función de regresión lineal
        R2 = regresion_lineal(columna_x, columna_y)
    
        # Guardar el valor de R2 en el diccionario con la columna como clave
        valores_R2[columna_x] = R2
    
        # Mostrar resultados por columna
        print(f"Regresión lineal para '{columna_x}' vs '{columna_y}':")
        print(f"Coeficiente de determinación (R^2): {R2}\n")
    
    except KeyError:
        print(f"Error: La columna '{columna_x}' no fue encontrada en el archivo.")
    except Exception as e:
        print(f"Error en la columna '{columna_x}': {e}")

# Mostrar todos los valores de R2
print("Valores de R2 por cada columna:")
print(valores_R2)
