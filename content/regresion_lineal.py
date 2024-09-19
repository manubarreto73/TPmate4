import numpy as np

def regresion_lineal(X, y):

    # Cálculo de las medias
    X_prom = np.mean(X)
    y_prom = np.mean(y)
    
    # Cálculo de las varianzas
    sxy = np.sum((X - X_prom) * (y - y_prom))
    sxx = np.sum((X - X_prom)**2)
    syy = np.sum((y - y_prom)**2)

    # Pendiente (β1)
    beta_1 = sxy / sxx
    
    # Intersección con el eje y (β0)
    beta_0 = y_prom - beta_1 * X_prom
    
    # Predicciones
    y_pred = beta_0 + beta_1 * X
    
    # Cálculo del coeficiente de correlación lineal
    R  = sxy / ((sxx * syy) ** 0.5)

    # Cálculo del coeficiente de determinación (R^2)
    # Este calculo no se usa porque terminé devolviendo R que es lo mismo, pero sirvió para corroborar que den lo mismo
    STC = np.sum((y - y_prom)**2)  # Suma total de cuadrados
    SCE = np.sum((y - y_pred)**2)  # Suma de los cuadrados de los errores
    R2 = 1 - (SCE / STC)

    #Calculos intervalos de confianza
    print(f"El intervalo para b0 es [ {beta_0 - (-1.96009) * ((SCE/18942 * (1/len(X) + (X_prom**2) / sxx) ) ** 0.5)}, {beta_0 + (-1.96009) * ((SCE/18942 * (1/len(X) + (X_prom**2) / sxx) ) ** 0.5)} ]")
    print(f"El intervalo para b1 es [ {beta_1 - (-1.96009) * ((((SCE/18942)**2) / sxx) ** 0.5)} , {beta_1 + (-1.96009) * ((((SCE/18942)**2) / sxx) ** 0.5)} ]")

    return R