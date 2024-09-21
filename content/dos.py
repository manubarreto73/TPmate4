import numpy as np
import pandas as pd

# 1. Cargar el dataset (supongamos que tienes el archivo players_21.csv en el mismo directorio)
data = pd.read_csv('players_21.csv')

# 2. Seleccionar las características relevantes (variables independientes)
# Por ejemplo: velocidad, pase, físico y disparo
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

# Hasta este punto, hemos preparado la matriz X, donde la primera columna es de unos (para el intercepto β0),
# y las demás columnas son las características.

# 6. Calcular los coeficientes usando la fórmula de mínimos cuadrados
# Fórmula: β = (X^T X)^-1 X^T y
X_transpose = X.T  # Transpuesta de X
beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y  # Cálculo de los coeficientes β

# - X^T es la transpuesta de la matriz X.
# - (X^T X)^-1 es la inversa del producto de X^T y X.
# - El producto de esta inversa con X^T y luego con el vector y nos da los coeficientes óptimos que minimizan los errores.

# 7. Mostrar los coeficientes obtenidos
print("Coeficientes (β):")
print(f"Intercepto (β0): {beta[0]}")
for i, feature in enumerate(features):
    print(f"Coeficiente para {feature} (β{i+1}): {beta[i+1]}")

# Hemos calculado el intercepto (β0) y los coeficientes (β1, β2, ...) correspondientes a cada variable independiente.
# Cada coeficiente indica cuánto cambia el valor de mercado por cada unidad de cambio en esa característica.

# 8. Calcular las predicciones (y_pred)
y_pred = X @ beta  # Producto matricial entre X y los coeficientes β

# **Explicación**:
# La predicción se realiza multiplicando la matriz X (que contiene las características de cada jugador)
# por el vector de coeficientes β. Esto nos da las predicciones de los valores de mercado para cada jugador.

# 9. Calcular el coeficiente de determinación (R²)
STC = np.sum((y - np.mean(y)) ** 2)  # Suma total de cuadrados
SCE = np.sum((y - y_pred) ** 2)  # Suma de cuadrados residuales
R2 = 1 - (SCE / STC)  # Cálculo de R²

print(f"\nCoeficiente de determinación (R²): {R2:.4f}")

# 10. Calcular la correlación entre y y y_pred
correlation_matrix = np.corrcoef(y, y_pred)
correlation = correlation_matrix[0, 1]  # Obtener la correlación de la matriz
print(f"Correlación entre valores predichos y reales: {correlation:.4f}")

# 11. Mostrar las primeras 5 predicciones junto a los valores reales
print("\nPredicciones y valores reales:")
for i in range(10):
    print(f"Jugador {i+1}: Valor predicho: {y_pred[i]:.2f}, Valor real: {y[i]:.2f}")

