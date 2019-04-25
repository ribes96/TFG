# Slide 2

Para este proyecto hemos querido tratar 2 de los aspectos que son un poco problemáticos dentro del Machine Learning. Estos aspectos son las diferencias entre los modelos lineales con los no-lineales, y el tema de los métodos ensemble.

La diferencia entre un modelo lineal y uno no-lineal es un poco técnica, y requiere un poco de tiempo explicarla. Si es necesario la puedo explicar más tarde. Pero, a grandes rasgos se puede hacer la siguiente diferenciación.

Los modelos lineales utilizan procedimientos sencillos para encontrar relaciones lineales entre las features de las instancias. Estos procedimientos son relativamente rápidos de ejecutar, y permiten reflejar correctamente algunos de los problemas de la vida real. Pero en la mayoría de los casos, estos modelos no tienen la suficiente flexibilidad para reflejar las relaciones entre las features de los problemas. Para ello se requiere usar modelos no-lineales, es decir, modelos que encuentran relaciones no-lineales en los datos. El problema es que estos procedimientos suelen ser mucho más costosos de ejecutar, y en algunos casos puede no ser viable usarlos, como es en el caso de las Support Vector Machines con algunos kernels.

El problema con los ensemble methods es el siguiente. Una forma de conseguir incrementar la precisión de un modelo de Machine Learning es entrenar a un grupo de estimadores en vez de a uno solo. Así, para predecir el resultado de una instancia, cada estimador realiza una predicción y se elige la predicción que haya recibido la mayoría de votos, en el caso de clasificación, o se hace la media de todos los estimadores, en el caso de regresión.

Esto es parecido a cuando se hace un comité de personas para tomar una decisión. Se espera que combinando el conocimiento de todos ellos la respuesta se ajuste mejor a la realidad. Sin embargo, es necesario que haya una cierta discrepancia entre los estimadores del ensemble. Si no la hay, es como si todos los expertos opinan de la misma manera, y el resultado global es el mismo que si tuviéramos un solo experto.

Es por este motivo que en los ensembles que usan el mismo modelo para todos los estimadores es necesario usar algoritmos que sean muy inestables. Por inestables queremos decir que con una pequeña diferencia en los datos de entrada se producen estimadores muy distintos.

En la práctica, el Decision Tree es de los pocos algoritmos que se benefician de este tipo de ensembles, pues el resto de modelos son demasiado estables. El algoritmo típico que usa un ensemble de Decision Tree es el Random Forest.


Pues, en este proyecto hemos intentado dar solución a estos dos problemas. Por una lado, hemos buscado la forma de usar modelos lineales, con las ventajas que tienen, para modelar problema no-lineales, y también hemos intentado desarrollar formas viables de hacer ensembles con otros modelos aparte del Decision Tree.

# Slide 3

Para hacerlo, usamos una aproximación del feature space de una función kernel.

Una función kernel es una función que equivale al dot product de sus entradas transformadas con alguna función, que llamaremos phi. Estas funciones se usan principalmente con las Support Vector Machine. El motivo es que estos modelos no utilizan directamente las entradas, sino que usan el dot product de sus entradas.

Una SVM (en el caso de clasificación) resuelve el problema de encontrar un híper-plano que separe las instancias en dos subconjuntos, uno para cada clase. El problema viene cuando no existe un plano que separe correctamente las instancias. En estos casos, una metodología típica es usar algún tipo de transformación de los datos que permita encontrar un híper-plano separador.

Las funciones kernel son ideales en estos casos, porque realizan una transformación implícita de los datos, de modo que no es necesario calcularla directamente. Una función kernel que se utiliza con mucha frecuencia y que es ideal para facilitar la separación de los datos es la llamada Radial Basis Function, RBF. Esta función tiene la particularidad de que su función implícita, phi, lleva los datos a un feature space de infinitas dimensiones, de modo que no es computable. Pero como las SVM no necesitan computar directamente esta función, esto no supone un problema.

Sin embargo, el uso del kernel trick tiene un par de inconvenientes. Por un lado, únicamente se puede usar con métodos que únicamente trabajen con los dos product de sus entradas, no con ellas directamente. Por otro lado, el uso de estas funciones hace que el problema de optimización de las SVM se vuelva mucho más costoso: tiene un coste cúbico con la cantidad de instancias. Este coste es inaceptable para large scale datasets.

A pesar de ello, existen algunas alternativas. Es posible generar una transformación aleatoria de los datos (que llamaremos z) que aproxima el valor de la función phi. Esta aproximación tiene poca dimensionalidad, y puede tener una precisión arbitraria. En este proyecto usamos dos formas de generar esta función: las Random Fourier Features y el método Nyström. La explicación es un poco técnica, y no la voy a explicar aquí.

# Slide 4

La propuesta de este trabajo es usar estos Random Mappings de dos formas distintas. Por una lado, queremos estudiar si esta transformación, que no es lineal, se puede usar para aprender las relaciones de problemas no-lineales con modelos lineales, con las ventajas que estos tienen. Por otro lado, puesto que nos permiten generar varias transformaciones distintas de un mismo dataset (todas ellas igual de válidas), queremos estudiar si nos podrían permitir usar los métodos ensemble con otros modelos aparte de los Decision Tree.

Hemos querido comprobar estas hipótesis con estos 3 modelos: Regresión Logística, Support Vector Machine y Decision Tree.

El uso de estas transformaciones sumará un coste adicional al entrenamiento de estos modelos. Sin embargo, este coste es lineal con la cantidad de instancias, de modo que debería ser escalable para large scale datasets.

# Slide 5

Respecto a la forma de combinar los Random Mapping con el método bagging, hay muchas formas de hacerlo. Nosotros nos hemos basado en el Random Forest para desarrollar 4 formas distintas de hacerlo. Inicialmente, sosteníamos la hipótesis de que una de estas formas sería más adecuada que las demás para realizar esta mezcla, pero finalmente todas han tenido unos resultados muy parecidos en los experimentos. Por eso mismo no las voy a explicar aquí.

# Slide 6

Las SVM pueden usar el kernel trick para resolver problemas no lineales. Sin embargo, esto hace que el coste de entrenarla sea cúbico con la cantidad de instancias, algo que es inaceptable con large scale datasets.

Con estos experimentos queríamos ver si es posible incrementar la precisión de una SVM lineal mediante el uso de los Random Mappings, pero con un coste mucho inferior al de usar el kernel trick.

Para hacer las pruebas hemos usado 9 datasets distintos, de los cuales 2 tienen muchas instancias, mientras que los otros tienen pocas. Por este motivo, solo se puede observar una reducción del tiempo de entrenamiento en los datasets grandes. Con los más pequeños, el coste añadido lineal es más grande que el coste cúbico de usar el kernel trick.

# Slide 7



# Slide 8
# Slide 9
# Slide 10
# Slide 11
