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

Se consiguieron los siguientes valores:
Coeficiente de determinación (R²): 0.3609
Correlación entre valores predichos y reales: 0.6007

El coeficiente de determinacion R2 indica la proporcion de la varianza de la variable dependiente "value_eur", es decir, el valor del mercado. En este caso un valor de 0.3609 indica que el 36.09% de la variabilidad en el valor de mercado de los jugadores puede ser explicada por las caracteristicas seleccionadas en el modelo. Es decir, el 63.91% de la variabilidad en el valor del mercado NO esta explicada por el modelo. Esto sugiere que el modelo es capaz de capturar una gran porcion de las variaciones en el valor de mercado pero que aun hay una cantidad considerable que el modelo no explica y que por lo tanto las caracteristicas elegidas no capturan las variables que influyen el valor de mercado, es decir, se podrian considerar mas u otras variables para mejorar el modelo.

por otro lado la correlacion entre valores predichos y reales mide la relacion lineal entre las predicciones del modelo y los valores reales. Un valor del 0.6007 indica una correlacion moderada entre los valores predichos y los valores reales, es decir que el modelo tiene capacidad predictiva pero hay margen para mejorar.
