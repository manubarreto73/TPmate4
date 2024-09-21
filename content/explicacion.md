Punto Uno A:
Usando Python, escribimos un programa para calcular una regresión lineal simple para cada una de las caracteristicas encontradas en el dataset brindado y asi obtener el coeficiente de determinacion (R2) de cada una de las regresiones para posteriormente poder compararlas y quedarnos con la regresion lineal cuyo coeficiente de determinacion fuera el mayor.
El R2 nos indica que tan bien el modelo ajusta los datos, el coeficiente oscila entre 0 y 1, donde un valor mas cercano a 1 sugiere que el modelo explica gran parte de la variabilidad en el valor de mercado a partir de la caracteristica seleccionada.

Prueba de Significancia de Regresión:
Realizamos esta prueba para verificar si la relación entre la característica seleccionada (reacciones de movimiento) y el valor de mercado es estadísticamente significativa. Nuestro objetivo es probar la hipotesis nula de que B0 es igual a 0. Esto implica verificar que la pendiente de la recta de regresión es diferente de cero. Es importante para respaldar la seleccion de la caracteristica mas relevante rechazar esta hipotesis, ya que de no hacerlo, cabe la posibilidad de que nuestra variable seleccionada como la mas determinante y el precio del jugador sean en realidad independientes entre si. Si el valor p asociado al estadístico de prueba es pequeño (generalmente menor a 0.05), rechazamos la hipótesis nula que afirma que no hay relación entre las variables. En nuestro caso, la prueba mostró que la característica seleccionada tiene una influencia significativa sobre el valor de mercado.

En resumen, estos elementos indican que la característica elegida es un predictor relevante y confiable del valor de mercado, con un ajuste razonable del modelo al comportamiento real de los datos. (podria hacer falta meter info sobre R)

Punto Uno C:

Dado que el enunciado pide buscar la proporción considerando ancho mínimo, consideramos un x* igual al promedio de los x, haciendo que el término donde se restan sea igual a 0, y en consecuencia minimizando el valor del error. Calculamos la proporción dividiendo el error de la predicción para futuros valores de x por el error de la respuesta media.