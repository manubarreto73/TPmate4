
import numpy as np
import pandas as pd

# 1. Cargar el dataset (supongamos que tienes el archivo players_21.csv en el mismo directorio)
data = pd.read_csv('players_21.csv')

# 2. Seleccionar las características relevantes (variables independientes)
features = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
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

# 3. Definir la variable objetivo (el valor de mercado)
target = 'value_eur'

# Asegúrate de que los datos no tengan valores nulos
data_cleaned = data[features + [target]].dropna()

# 4. Preparar las matrices X (características) y y (valores objetivo)
X = data[features].values  # Matriz de variables independientes
y = data[target].values  # Vector de valores de mercado

# 5. Agregar una columna de unos a X para el intercepto (β0)
ones_column = np.ones((X.shape[0], 1))
X = np.concatenate((ones_column, X), axis=1)

# 6. Definir parámetros para el descenso por gradiente
learning_rate = 0.00001  # Tasa de aprendizaje (ajustar según el caso)
tolerance = 0.001 # Tolerancia para detener el algoritmo
max_iterations = 1000000  # Número de iteraciones
n_samples, n_features = X.shape  # Número de muestras y características

# 7. Inicializar los coeficientes (pesos) β
beta = [-128848.47820192, 7183.22539255, 14080.46941196, -16201.09749207, -1942.95569367, 28211.71259667, 23277.10896274, 29419.78868596, -20517.18969699, 6375.30083468, 9756.72945743, -4020.83758484, 20465.45713952, -56224.38121119, 203613.79733419, -88302.70088258, 9557.04889458, -7128.9767184, 6660.84499929, -101178.01839044, -8572.99486205, -5510.30717994, 13851.41266295, -19795.68040287, 9373.20128304, -48715.38719479, 78996.79344382, 17376.51754433, -38400.85059305, -4580.4537823, -8248.89440088, -17917.9821524, -10517.0220651, 3885.29764952]

# 8. Algoritmo de descenso por gradiente
for iteration in range(max_iterations):
    # Calcular las predicciones actuales
    y_pred = X @ beta

    # Calcular el gradiente de la función de costo
    gradient = (1 / n_samples) * X.T @ (y_pred - y)
    
    # Actualizar los coeficientes
    beta_new = beta - learning_rate * gradient
    
    # Calcular el cambio en los coeficientes
    beta_change = np.linalg.norm(beta_new - beta)

    # Actualizar los coeficientes para la siguiente iteracion
    beta = beta_new

    # Mostrar las últimas 5 iteraciones y el cambio en los coeficientes
    if iteration >= max_iterations - 5 or beta_change < tolerance:
        print(f"Iteración {iteration + 1}/{max_iterations}:")
        print(f"Coeficientes actuales: {beta}")
        print(f"Cambio en los coeficientes: {beta_change}")

    # Parar si el cambio en los coeficientes es menor a la tolerancia
    if beta_change < tolerance:
        print(f"\nEl algoritmo ha convergido después de {iteration + 1} iteraciones.")
        break# 9. Mostrar los coeficientes obtenidos al finalizar


print("\nCoeficientes finales (β):")
print(f"Intercepto (β0): {beta[0]}")
for i, feature in enumerate(features):
    print(f"Coeficiente para {feature} (β{i+1}): {beta[i+1]}")

# 10. Calcular las predicciones finales
y_pred_final = X @ beta

# 11. Calcular el coeficiente de determinación (R²) para evaluar el modelo
ss_total = np.sum((y - np.mean(y)) ** 2)  # Suma total de cuadrados
ss_residual = np.sum((y - y_pred_final) ** 2)  # Suma de cuadrados residuales
r_squared = 1 - (ss_residual / ss_total)  # Cálculo de R²

print(f"\nCoeficiente de determinación (R²): {r_squared:.4f}")
