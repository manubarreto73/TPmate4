import numpy as np

def regresion_lineal(atributo, X, y):

    # Cálculo de las medias
    X_prom = np.mean(X)
    y_prom = np.mean(y)
    
    # Cálculo de las varianzas
    sxy = np.sum((X - X_prom) * (y - y_prom))
    sxx = np.sum((X - X_prom)**2)
    syy = np.sum((y - y_prom)**2)

    # Cánculo de la pendiente (β1) y la ordenada al origen(β0)
    beta_1 = sxy / sxx
    beta_0 = y_prom - beta_1 * X_prom
    
    # Cálculo del coeficiente de correlación lineal
    R  = sxy / ((sxx * syy) ** 0.5)

    # Predicciones
    y_pred = beta_0 + beta_1 * X

    # Cálculo del coeficiente de determinación (R^2)
    # Este calculo no se utiliza finalmente, ya que devolvemos R y lo elevamos al cuadrado
    STC = np.sum((y - y_prom)**2)  # Suma total de cuadrados
    SCE = np.sum((y - y_pred)**2)  # Suma de los cuadrados de los errores
    R2 = 1 - (SCE / STC) #Calculo de R2

    #Incisos ii y iii de la parte 1, solo resueltos para el atributo más significativo
    if (atributo == 'movement_reactions'):
        #Intervalos de confianza para b0 y b1
        varianza = SCE / 18942
        estadistico_prueba = beta_1 / ((varianza / sxx) ** 0.5)
        print(f"El intervalo para b0 es [ {beta_0 - (1.96009) * ((varianza * (1/len(X) + (X_prom**2) / sxx) ) ** 0.5)}, {beta_0 + (1.96009) * ((varianza * (1/len(X) + (X_prom**2) / sxx) ) ** 0.5)} ]")
        print(f"El intervalo para b1 es [ {beta_1 - (1.96009) * ((varianza / sxx) ** 0.5)} , {beta_1 + (1.96009) * ((varianza / sxx) ** 0.5)} ]")
        #Proporcion entre el error de predicción y el de confianza, considerando ancho mínimo
        error_de_prediccion = 1.96009 * ((varianza * (1 + 1/len(X))) ** 0.5)
        error_de_confianza = 1.96009 * ((varianza * (1/len(X))) ** 0.5)
        print(f"La proporción entre el error de predicción y el de confianza {error_de_prediccion / error_de_confianza}")

    return R