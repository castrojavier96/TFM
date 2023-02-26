# TFM

Este proyecto es un rebalanceador de estrategias de inversion basado en aprendizaje supervisado, se define el modelo principal junto con dos modelos de prueba 
y se entregan predicciones segun los inputs, se realiza tambien un backtesting y se comparan todas las soluciones junto con un benchmark sintetico y una prueba de aleatoriedad.

El proposito es realizar un proyecto integro que considere todas las aristas, implementado con una solucion cloud, con backtesting y distintas pruebas para testear si el modelo aprende.

El script principal es el rebalanceador.py, desde ese script se instalan todas las librerias necesarias leyendo el requeriments.txt y posteriormente llama a todas las funciones definidas en los otros scripts. por lo que solo hay que ejecutar el script rebalanceador.py.

una consideracion es que la prueba de aleatoriedad puede tardar mucho, por lo que se dejo solo con 10 monos, aunque para los resultados y la memoria se hizo con 2000 monos.