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

ii)

los valores obtenidos mediante el metodo de descenso por gradiente NO son iguales a los conseguidos mediante la resolucion del sistema de ecuaciones normales. 

*inserte resultados*

Iteración 999996/1000000:
[-1.36648236e+06  6.18143186e+03  1.44102634e+04 -1.37764196e+04
  5.92336994e+02  2.75868679e+04  2.34582913e+04  2.88502808e+04
 -1.98537234e+04  5.51155190e+03  1.17762698e+04 -2.38009249e+03
  2.13726821e+04 -5.53353667e+04  2.03342826e+05 -8.32887167e+04
  9.85750178e+03 -6.88083458e+03  6.32036152e+03 -9.62419926e+04
 -1.03563233e+04 -6.06131534e+03  1.18031333e+04 -2.01396234e+04
  1.01854870e+04 -4.68555979e+04  7.70514140e+04  1.87590332e+04
 -3.65877145e+04 -2.87940937e+03 -5.71922055e+03 -1.58912224e+04
 -1.00116017e+04  5.40897814e+03]
Cambio en los coeficientes: 1.203014196951658
Iteración 999997/1000000:
[-1.36648357e+06  6.18143089e+03  1.44102637e+04 -1.37764172e+04
  5.92339458e+02  2.75868673e+04  2.34582914e+04  2.88502802e+04
 -1.98537227e+04  5.51155106e+03  1.17762718e+04 -2.38009090e+03
  2.13726829e+04 -5.53353658e+04  2.03342826e+05 -8.32887118e+04
  9.85750207e+03 -6.88083434e+03  6.32036119e+03 -9.62419878e+04
 -1.03563250e+04 -6.06131588e+03  1.18031313e+04 -2.01396237e+04
  1.01854878e+04 -4.68555961e+04  7.70514121e+04  1.87590345e+04
 -3.65877128e+04 -2.87940772e+03 -5.71921809e+03 -1.58912204e+04
 -1.00116012e+04  5.40897962e+03]
Cambio en los coeficientes: 1.2030141289626433
Iteración 999998/1000000:
[-1.36648477e+06  6.18142991e+03  1.44102640e+04 -1.37764149e+04
  5.92341923e+02  2.75868667e+04  2.34582916e+04  2.88502797e+04
 -1.98537221e+04  5.51155022e+03  1.17762737e+04 -2.38008931e+03
  2.13726838e+04 -5.53353649e+04  2.03342826e+05 -8.32887069e+04
  9.85750236e+03 -6.88083410e+03  6.32036086e+03 -9.62419830e+04
 -1.03563268e+04 -6.06131641e+03  1.18031293e+04 -2.01396240e+04
  1.01854886e+04 -4.68555943e+04  7.70514102e+04  1.87590359e+04
 -3.65877110e+04 -2.87940607e+03 -5.71921563e+03 -1.58912184e+04
 -1.00116007e+04  5.40898111e+03]
Cambio en los coeficientes: 1.2030140609736149
Iteración 999999/1000000:
[-1.36648597e+06  6.18142894e+03  1.44102643e+04 -1.37764125e+04
  5.92344387e+02  2.75868661e+04  2.34582918e+04  2.88502791e+04
 -1.98537214e+04  5.51154938e+03  1.17762757e+04 -2.38008771e+03
  2.13726847e+04 -5.53353641e+04  2.03342826e+05 -8.32887021e+04
  9.85750265e+03 -6.88083385e+03  6.32036053e+03 -9.62419782e+04
 -1.03563285e+04 -6.06131695e+03  1.18031273e+04 -2.01396244e+04
  1.01854894e+04 -4.68555925e+04  7.70514083e+04  1.87590372e+04
 -3.65877092e+04 -2.87940441e+03 -5.71921317e+03 -1.58912165e+04
 -1.00116002e+04  5.40898259e+03]
Cambio en los coeficientes: 1.2030139927518095
Iteración 1000000/1000000:
[-1.36648718e+06  6.18142797e+03  1.44102647e+04 -1.37764102e+04
  5.92346852e+02  2.75868655e+04  2.34582920e+04  2.88502786e+04
 -1.98537208e+04  5.51154855e+03  1.17762776e+04 -2.38008612e+03
  2.13726856e+04 -5.53353632e+04  2.03342825e+05 -8.32886972e+04
  9.85750295e+03 -6.88083361e+03  6.32036020e+03 -9.62419734e+04
 -1.03563302e+04 -6.06131749e+03  1.18031253e+04 -2.01396247e+04
  1.01854901e+04 -4.68555907e+04  7.70514064e+04  1.87590386e+04
 -3.65877075e+04 -2.87940276e+03 -5.71921071e+03 -1.58912145e+04
 -1.00115997e+04  5.40898407e+03]
Cambio en los coeficientes: 1.2030139247627927

Coeficientes finales (β):
Intercepto (β0): -1366487.1755963096
Coeficiente para attacking_crossing (β1): 6181.427965966624
Coeficiente para attacking_finishing (β2): 14410.26466388734
Coeficiente para attacking_heading_accuracy (β3): -13776.41016008651
Coeficiente para attacking_short_passing (β4): 592.3468517149188
Coeficiente para attacking_volleys (β5): 27586.865496276776
Coeficiente para skill_dribbling (β6): 23458.291967332516
Coeficiente para skill_curve (β7): 28850.278582059924
Coeficiente para skill_fk_accuracy (β8): -19853.72078945289
Coeficiente para skill_long_passing (β9): 5511.548545252358
Coeficiente para skill_ball_control (β10): 11776.277640240789
Coeficiente para movement_acceleration (β11): -2380.0861157350587
Coeficiente para movement_sprint_speed (β12): 21372.685595199073
Coeficiente para movement_agility (β13): -55335.36319925679
Coeficiente para movement_reactions (β14): 203342.82544476658
Coeficiente para movement_balance (β15): -83288.69719423205
Coeficiente para power_shot_power (β16): 9857.502945444963
Coeficiente para power_jumping (β17): -6880.83361381116
Coeficiente para power_stamina (β18): 6320.360196372235
Coeficiente para power_strength (β19): -96241.97337605184
Coeficiente para power_long_shots (β20): -10356.330230652993
Coeficiente para mentality_aggression (β21): -6061.31748528832
Coeficiente para mentality_interceptions (β22): 11803.125331733054
Coeficiente para mentality_positioning (β23): -20139.62469135874
Coeficiente para mentality_vision (β24): 10185.490148711953
Coeficiente para mentality_penalties (β25): -46855.590665957265
Coeficiente para mentality_composure (β26): 77051.40644714198
Coeficiente para defending_standing_tackle (β27): 18759.03856618369
Coeficiente para defending_sliding_tackle (β28): -36587.70748504327
Coeficiente para goalkeeping_diving (β29): -2879.402757287389
Coeficiente para goalkeeping_handling (β30): -5719.210709538186
Coeficiente para goalkeeping_kicking (β31): -15891.214506630393
Coeficiente para goalkeeping_positioning (β32): -10011.59971215418
Coeficiente para goalkeeping_reflexes (β33): 5408.984065332548

Coeficiente de determinación (R²): 0.2626
Los parametros utilizados fueron:

tasa de aprendizaje = 0.00001
tolerancia = 0.001
limite de iteraciones = 1000000

en nuestro caso incluso luego de 1000000 de iteraciones, no se llego a converger en la tolerancia.


iii)
El criterio de corte definido para el algoritmo de descenso por gradiente es la norma de cambio entre los coeficientes entre una iteración y otra. En nuestro caso, el algoritmo se detendrá si y solo si el cambio entre los coeficientes es menor a una tolerancia predefinida de 0.001, o hasta que se llegue al número máximo de iteraciones de un millon. Existen fallas potenciales en el criterio. En primer lugar, si la tasa de aprendizaje es demasiado grande, los coeficientes pueden oscilar en lugar de converger, si esto sucede, el algoritmo puede no detenerse nunca, o lo puede hacer antes de tiempo en el valor incorrecto. De la misma forma, si la tasa de aprendizaje es muy pequeña y no se ha llegado al minimo, el criterio basado en el cambio de los coeficientes puede detener el algoritmo antes de tiempo. Un criterio mejor sería el basado en la funcion de costo. En lugar de verificar el cambio en los coeficientes, vemos la diferencia en el costo (por ejemplo el error cuadrático medio), entre iteraciones. Por lo tanto, si la diferencia en el costo es menor que la tolerancia, paramos el algoritmo. Esto garantiza que el modelo no se detenga hasta que el error sea lo suficientemente cercano a cero, independientemente del cambio en los coeficientes.

