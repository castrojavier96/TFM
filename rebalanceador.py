# REBALANCEADOR
# Script principal que llama todas las funciones y muestra los resultados 

#Se instalan todas las librerias a utilizar
import subprocess

# Definir el comando que se ejecutará en la terminal
command = 'pip install -r requirements.txt'

# Ejecutar el comando en la terminal desde Python
subprocess.call(command, shell=True)

# Se importan las librerias previamente instaladas
import numpy as np
import pandas as pd
from estrategia_1_Momentum import obtener_precios_mom
#from estrategia_2_HRP import obtener_precios_HRP
from estrategia_3_EW import obtener_precios_EW
from estrategia_4_Volatilidad import obtener_precios_vol
from estrategia_5_MINVolatilidad import obtener_precios_volmin
from benchmark import benchmark
from monos import monos
from datos_s3 import obtener_datos, obtener_macros
from modelo_rebalanceador import modelo
from backtesting import backtesting
from regresion_logistica import regresion_logistica
from red_neuronal import red_neuronal
import matplotlib.pyplot as plt
from graficos import graficar_series_precio
from scipy.stats import percentileofscore

# Se obtienen los datos y se definen los parámetros iniciales
activos_close, activos_volumen, activos_open = obtener_datos()
inflacion = obtener_macros()
n_activos = 10
capital_inicial = 10000
com = 0.0003
dias_reb = 7

# Se calcula la rentabilidad de los activos y se visualizan las rentabilidades
df_retornos = activos_close.pct_change()
df_rentabilidad = (1 + df_retornos).cumprod()

# Obtenemos los 5 activos con mayor rentabilidad acumulada al final del periodo
activos_top5 = df_rentabilidad.iloc[-1].sort_values(ascending=False)[:5].index.tolist()
activos_bot5 = df_rentabilidad.iloc[-1].sort_values(ascending=True)[:5].index.tolist()
activos_top_bot = lista3 = activos_top5 + activos_bot5

# Graficamos la rentabilidad acumulada de los 5 activos con mayor y menor rentabilidad acumulada
fig, ax = plt.subplots(figsize=(10, 5))
df_rentabilidad[activos_top_bot].plot(ax=ax)

# Creamos la leyenda para los 5 primeros y los 5 últimos activos
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [label for label in labels if label in activos_top5 or label in activos_bot5]
plt.legend(handles, new_labels, loc='upper left')
plt.title('Rentabilidad acumulada de los 5 primeros y los 5 últimos activos')
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad acumulada')
plt.grid(True)
plt.show()

dias_reb = 15

# Se calculan las series de precios de cada estrategia y se obtienen los datos macroeconómicos
serie_momentum = obtener_precios_mom(activos_close, n_activos, capital_inicial, com, dias_reb)
serie_EW = obtener_precios_EW(activos_close, activos_volumen, n_activos, capital_inicial, com, dias_reb)
serie_volatilidad = obtener_precios_vol(activos_close, n_activos, capital_inicial, com, dias_reb)
serie_volmin = obtener_precios_volmin(activos_close, n_activos, capital_inicial, com, dias_reb)

graficar_series_precio(serie_momentum, serie_EW, serie_volatilidad, serie_volmin)

# MODELO (SE DEFINE Y ENTRENA NUESTRO MODELO, Y NOS QUEDAMOS CON EL OUTPUT Y LAS PREDICCIONES)
model, datos_inputs, datos_output, datos_inputs_esc = modelo(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, inflacion, dias_reb)
y_hat = model.predict(datos_inputs_esc)

# REGRESIÓN LOGÍSTICA (SE DEFINE Y ENTRENA UNA REGRESIÓN LOGÍSTICA, Y NOS QUEDAMOS CON EL OUTPUT Y LAS PREDICCIONES)
modelo_regresion = regresion_logistica(datos_inputs, datos_output)
y_hat_regresion = modelo_regresion.predict(datos_inputs_esc)

# RED NEURONAL (SE DEFINE Y ENTRENA UN MODELO SIMPLE DE REDES, Y NOS QUEDAMOS CON EL OUTPUT Y LAS PREDICCIONES)
modelo_redneuronal = red_neuronal(datos_inputs, datos_output)
y_hat_redneuronal = modelo_redneuronal.predict(datos_inputs_esc)
y_hat_redneuronal = y_hat_redneuronal.numpy()

# BACKTESTING DE NUESTRO MODELO
serie_backtesting, comision_backtesting, diferencias, ratio_sharpe_modelo, ratio_sortino_modelo, drawdown, maxdradown_modelo = backtesting(activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, y_hat, dias_reb)

# Calculamos los retornos acumulados y guardamos un Dataframe con todas las ordenes de compra
df_retornos = serie_backtesting.pct_change()
serie_backtesting = (1 + df_retornos).cumprod()
df_diferencias = pd.concat(diferencias, axis=1).transpose()

# Se grafica la rentabilidad acumulada, las predicciones, el drawdown y los resultados
plt.figure()
plt.plot(serie_backtesting, label='Backtesting Modelo')
plt.title('Backtesting Modelo')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()

plt.figure()
plt.plot(y_hat)
plt.title('Predicciones Modelo')
plt.xlabel('meses')
plt.ylabel('predicciones')

plt.figure()
plt.plot(drawdown, label='Drawdown Modelo')
plt.title('Drawdown Modelo')
plt.xlabel('Fecha')
plt.ylabel('Porcentaje')
plt.legend()

print(f"La rentabilidad acumulada del modelo es: {round(serie_backtesting.iloc[-1]*100, 2)}%")
print(f"La comision total del modelo es: {comision_backtesting}")
print(f"El ratio de Sharpe del modelo es: {ratio_sharpe_modelo}")
print(f"El ratio de Sortino del modelo es: {ratio_sortino_modelo}")
print(f"El Maximo drawdown del modelo es: {maxdradown_modelo}")
print()

# BACKTESTING REGRESION LOGISTICA
serie_regresion, comision_regresion, diferencias_regresion, ratio_sharpe_reg, ratio_sortino_reg, drawdown, maxdradown_reg = backtesting(activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, y_hat_regresion, dias_reb)

# Calculamos los retornos acumulados y guardamos un DataFrame con todas las ordenes de compra
df_retornos = serie_regresion.pct_change()
serie_regresion = (1 + df_retornos).cumprod()
df_diferencias = pd.concat(diferencias, axis=1).transpose()
df_diferencias_regresion = pd.concat(diferencias_regresion, axis=1).transpose()

# Se grafica la rentabilidad acumulada, las predicciones, el drawdown y los resultados
plt.figure()
plt.plot(serie_regresion, label='Backtesting Modelo "Regresion Logistica"')
plt.title('Backtesting Regresion')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()

plt.figure()
plt.plot(y_hat_regresion)
plt.title('Predicciones Regresion Logistica')
plt.xlabel('epoch')
plt.ylabel('predicciones')

plt.figure()
plt.plot(drawdown, label='Drawdown Modelo Regresion Logistica')
plt.title('Drawdown Modelo Regresion Logistica')
plt.xlabel('Fecha')
plt.ylabel('Porcentaje')
plt.legend()

print(f"La rentabilidad acumulada del modelo Regresion Logistica es: {round(serie_regresion.iloc[-1] * 100, 2)}%")
print(f"La comision total del modelo Regresion Logistica es: {comision_regresion}")
print(f"El ratio de Sharpe del modelo Regresion Logistica es: {ratio_sharpe_reg}")
print(f"El ratio de Sortino del modelo Regresion Logistica es: {ratio_sortino_reg}")
print(f"El Maximo drawdown del modelo Regresion Logistica es: {maxdradown_reg}")
print()

#BACKTESTING RED NEURONAL
serie_red, comision_red, diferencias_red, ratio_sharpe_red, ratio_sortino_red, drawdown, maxdradown_red = backtesting(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, y_hat_redneuronal, dias_reb)

# calculamos los retornos acumulados y guardamos un Dataframe con todas las ordenes de compra
df_retornos = serie_red.pct_change()
serie_red = (1 + df_retornos).cumprod()
df_diferencias_red = pd.concat(diferencias_red, axis=1).transpose()

# Se grafica la rentabilidad acumulada, las predicciones, el dradown y los resultados
plt.figure()
plt.plot(serie_red, label = 'Backtesting Modelo "Red Neuronal"')
plt.title('Backtesting Red Neuronal')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()

plt.figure()
plt.plot(y_hat_redneuronal)
plt.title('Predicciones Red Neuronal')
plt.xlabel('epoch')
plt.ylabel('predicciones')

plt.figure()
plt.plot(drawdown, label = 'Drawdown Modelo Red Neuronal')
plt.title('Drawdown Modelo Red Neuronal')
plt.xlabel('Fecha')
plt.ylabel('Porcentaje')
plt.legend()

print(f"La rentabilidad acumulada del modelo Red Neuronal es: {round(serie_red.iloc[-1]*100,2)}%")
print(f"La comision total del modelo Red Neuronal es: {comision_red}")
print(f"El ratio de Sharpe del modelo Red Neuronal es: {ratio_sharpe_red}")
print(f"El ratio de Sortino del modelo Red Neuronal es: {ratio_sortino_red}")
print(f"El Maximo drawdown del modelo Red Neuronal es: {maxdradown_red}")

# BENCHMARK SINTETICO - EW de las estrategias
# Se crea un benchamk sintetico que hace todos los meses un Equal Wight de las estratgeias y el cash, se grafican los resultados
serie_benchmark, comision_benchmark, diferencias, ratio_sharpe_bench, ratio_sortino_bench, drawdown, maxdradown_bench = benchmark(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, dias_reb)

df_retornos = serie_benchmark.pct_change()
serie_benchmark = (1 + df_retornos).cumprod()
df_diferencias_red = pd.concat(diferencias_red, axis=1).transpose()

# Se grafica la rentabilidad acumulada, las predicciones, el dradown y los resultados
plt.figure()
plt.plot(serie_benchmark)
plt.title('Backtesting Benchmark')
plt.xlabel('Fecha')
plt.ylabel('Precio')

plt.figure()
plt.plot(drawdown, label = 'Drawdown Benchmark')
plt.title('Drawdown Benchmark')
plt.xlabel('Fecha')
plt.ylabel('Porcentaje')
plt.legend()

print(f"La rentabilidad acumulada del benchmark es: {round(serie_benchmark.iloc[-1]*100,2)}%")
print(f"La comision total del benchmark es: {comision_benchmark}")
print(f"El ratio de Sharpe del benchmark es: {ratio_sharpe_bench}")
print(f"El ratio de Sortino del benchmark es: {ratio_sortino_bench}")
print(f"El Maximo drawdown del benchmark es: {maxdradown_bench}")



# MONOS - PRUEBA DE ALEATORIEDAD   100 MONOS SE DEMORAN COMO 13 MINUTOS
serie_monos = []
comisiones_monos = []
for i in range(10):
    serie_monos_int, comision_monos, diferencias = monos(
        activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, dias_reb
    )
    serie_monos.append(serie_monos_int)
    comisiones_monos.append(comision_monos)
serie_monos = pd.DataFrame(serie_monos).transpose()
serie_monos.set_index(activos_close.index[42:], inplace=True)
comisiones_monos = pd.DataFrame(comisiones_monos).transpose()
plt.figure()
plt.plot(serie_monos)
plt.title('Backtesting Prueba Aleatoriedad')
plt.xlabel('Fecha')
plt.ylabel('Precio')

# Se calcula el mono 50 y el 100 para compararlos con los modelos
ranking_monos = serie_monos.iloc[-1, :].sort_values(ascending=False)
fila_central = (ranking_monos.shape[0] - 1) // 2
mono_50 = ranking_monos[ranking_monos.index[fila_central]]
comision_50 = comisiones_monos.iloc[:, ranking_monos.index[fila_central]]
mono_100 = ranking_monos[ranking_monos.index[0]]
comision_100 = comisiones_monos.iloc[:, ranking_monos.index[0]]

# Se calculan los retonos de los monos seleccionados 
# Mono 50%
df_retornos = serie_monos.iloc[:, ranking_monos.index[fila_central]].pct_change()
serie_monos.iloc[:, ranking_monos.index[fila_central]] = (1 + df_retornos).cumprod()
# Mono 100%
df_retornos = serie_monos.iloc[:, ranking_monos.index[0]].pct_change()
serie_monos.iloc[:, ranking_monos.index[0]] = (1 + df_retornos).cumprod()

valor = serie_backtesting.iloc[-1] * 44444  # Obtener el valor de la tercera columna en la primera fila
percentil = percentileofscore(ranking_monos, valor)
print(percentil)

# Se grafican las series de precios de los 3 modelos, el benchmark y el mono del 50%
plt.figure()
plt.plot(serie_benchmark, label="benchmark sintetico")
plt.plot(serie_red, label="Red Neuronal")
plt.plot(serie_regresion, label="Regresion logistica")
plt.plot(serie_backtesting, label="Modelo")
plt.plot(serie_monos.iloc[:, ranking_monos.index[fila_central]], label="Mono 50")
plt.plot(serie_monos.iloc[:, ranking_monos.index[0]], label="Mono 100")
plt.title('Backtesting')
plt.xlabel('fecha')
plt.ylabel('precio')
plt.legend()

# Se concatenan las comisiones de todo lo anterior y se grafican
plt.figure()
categorias = ['Modelo', 'Red Neuronal', 'Regresion', 'Benchmark', 'Mono 50%', 'Mono 100%']
valores = [comision_backtesting, comision_red, comision_regresion, comision_benchmark, comision_50.iloc[0], comision_100.iloc[0]]
colores = plt.cm.Set1(np.linspace(0, 1, len(categorias)))
fig, ax = plt.subplots()
ax.bar(categorias, valores, color=colores, width=0.1)
ax.tick_params(axis='x', labelsize=8)
plt.title('Comisiones')
plt.show()
