Hicimos un código que implementa una regresión lineal múltiple para predecir el valor de mercado de los jugadores en función de varias características extraídas del documento CSV. El objetivo del modelo es ajustar los valores de mercado de los jugadores en base a sus atributos individuales, utilizando el método de mínimos cuadrados.

Primero se carga el archivo players_21.csv que contiene la informacion sobre los jugadores. Esto se hace usando la libreria de Pandas para cargar el dataset. Se elijen las caracteristicas independientes utilizadas para precedir el valor de mercado de los jugadores (en nuestro caso utilizamos todas las que habian).
Nuestra variable objetivo es el valor en euros del jugador.

Se preparan las matrices de caracteristicas (X) y objetivo (Y). La matriz X contiene las caracteristicas seleccionadas de los jugadores, cada fila representa a un jugador y cada columna una caracteristica como "attacking_crossing" o "movement_acceleration". El vector Y es un vector unidimensional que contiene el valor de mercado para cada jugador.

Agregamos una columna de unos para el intercepto al comienzo de X, de esta manera el se puede inlcuir el intercepto B0 en la ecuacion de regresion. Si no agregaramos la columna de unos, el modelo no tendria un termino constante B0.  --> esto esta raro.

Luego calculamos los coeficientes usando la formula de minimos cuadrados:
B^=(XT*X)^−1*XTy

El resultado de esta operacion es un vector de coeficientes bn que incluye tanto el intercepto b0 como los coeficientes b1, b2.. bn correspondientes a cada una de las caracteristicas seleccionadas.

luego se calculan las predicciones del valor de mercado para cada jugador utilizando la ecuacion de regresion lineal multiple, el producto matricial X@B nos da un vector que contiene los valores de mercado predichos para cada jugador.

Al final solo se comparan los valores de las predicciones con los valores reales.
