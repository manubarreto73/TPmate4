
import numpy as np
import pandas as pd

data = pd.read_csv('players_21.csv')

caracteristicas = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 
    'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 
    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 
    'movement_agility', 'movement_reactions', 'movement_balance', 
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
    'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 
    'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
    'mentality_composure', 'defending_standing_tackle', 
    'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 
    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

target = 'value_eur'

X = data[caracteristicas] .values  # Matriz de variables independientes
y = data[target].values  # Vector de valores de mercado

# Agrego una columna de unos a X para el intercepto (β0)
ones_column = np.ones((X.shape[0], 1))
X = np.concatenate((ones_column, X), axis=1)

# Defino los parámetros para el descenso por gradiente
tasa_aprendizaje = 0.00001
tolerancia = 0.001
iteraciones_max = 1000000
n_muestras, n_caracs = X.shape

# Inicializar los coeficientes β (valores de la ultima vez que se ejecuto el algoritmo)
beta = [-128848.47820192, 7183.22539255, 14080.46941196, -16201.09749207, -1942.95569367, 28211.71259667, 23277.10896274, 29419.78868596, -20517.18969699, 6375.30083468, 9756.72945743, -4020.83758484, 20465.45713952, -56224.38121119, 203613.79733419, -88302.70088258, 9557.04889458, -7128.9767184, 6660.84499929, -101178.01839044, -8572.99486205, -5510.30717994, 13851.41266295, -19795.68040287, 9373.20128304, -48715.38719479, 78996.79344382, 17376.51754433, -38400.85059305, -4580.4537823, -8248.89440088, -17917.9821524, -10517.0220651, 3885.29764952]

# Algoritmo de descenso por gradiente
for iteracion in range(iteraciones_max):
    # Calculo de las predicciones actuales
    y_pred = X @ beta

    # Calculo del gradiente
    gradiente = (1 / n_muestras) * X.T @ (y_pred - y)
    
    # Actualizar los coeficientes
    beta_nuevo = beta - tasa_aprendizaje * gradiente
    
    # Calculo del cambio en los coeficientes
    cambio_beta = np.linalg.norm(beta_nuevo - beta)

    # Actualizacion de los coeficientes para la siguiente iteracion
    beta = beta_nuevo

    if iteracion >= iteraciones_max - 5 or cambio_beta < tolerancia:
        print(f"Iteración {iteracion + 1}/{iteraciones_max}:")
        print(f"Coeficientes actuales: {beta}")
        print(f"Cambio en los coeficientes: {cambio_beta}")

    # Parar si el cambio en los coeficientes es menor a la tolerancia
    if cambio_beta < tolerancia:
        print(f"\nEl algoritmo ha convergido después de {iteracion + 1} iteraciones.")
        break

# Mostramos los coeficientes obtenidos
print("\nCoeficientes finales (β):")
print(f"Intercepto (β0): {beta[0]}")
for i, carac in enumerate(caracteristicas):
    print(f"Coeficiente para {carac} (β{i+1}): {beta[i+1]}")

# Calculo de las predicciones finales
y_pred_final = X @ beta

# Calculo del coeficiente de determinación (R²) para evaluar el modelo
ss_total = np.sum((y - np.mean(y)) ** 2)  # Suma total de cuadrados
ss_residual = np.sum((y - y_pred_final) ** 2)  # Suma de cuadrados residuales
r_2 = 1 - (ss_residual / ss_total)  # Cálculo de R²

print(f"\nCoeficiente de determinación (R²): {r_2:.4f}")
