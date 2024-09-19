import numpy as np

def regresion_lineal(X, y):
    # Cálculo de los coeficientes de la recta de regresión (método de mínimos cuadrados)
    X_prom = np.mean(X)
    y_prom = np.mean(y)
    
    # Pendiente (β1)
    beta_1 = np.sum((X - X_prom) * (y - y_prom)) / np.sum((X - X_prom)**2)
    
    # Intersección con el eje y (β0)
    beta_0 = y_prom - beta_1 * X_prom
    
    # Predicciones
    y_pred = beta_0 + beta_1 * X
    
    # Cálculo del coeficiente de determinación (R^2)
    STC = np.sum((y - y_prom)**2)  # Suma total de cuadrados
    SCE = np.sum((y - y_pred)**2)  # Suma de los cuadrados de los errores
    R2 = 1 - (SCE / STC)
    
    return R2
