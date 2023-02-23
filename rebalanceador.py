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

# se obtienen los datps y se definen parametros iniciales
activos_close, activos_volumen, activos_open = obtener_datos()
n_activos = 10
capital_inicial = 10000
com = 0.0003

# Se calculan las series de precios de cada estrategia y se obtiene los datos macroeconomicos
serie_momentum = obtener_precios_mom(activos_close,n_activos,capital_inicial,com)
#serie_HRP = obtener_precios_HRP(activos_close)
serie_EW = obtener_precios_EW(activos_close,activos_volumen,n_activos,capital_inicial,com)
serie_volatilidad = obtener_precios_vol(activos_close,n_activos,capital_inicial,com)
serie_volmin  = obtener_precios_volmin(activos_close,n_activos,capital_inicial,com)
inflacion, pib = obtener_macros()

# MODELO
model, datos_inputs, datos_output, datos_inputs_esc= modelo(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, inflacion, pib)
y_hat = model.predict(datos_inputs_esc)

#REGRESION LOGISTICA
modelo_regresion = regresion_logistica(datos_inputs, datos_output)
y_hat_regresion = modelo_regresion.predict(datos_inputs_esc)

#RED NEURONAL
modelo_redneuronal = red_neuronal(datos_inputs, datos_output)
y_hat_redneuronal = modelo_redneuronal.predict(datos_inputs_esc)
y_hat_redneuronal = y_hat_redneuronal.numpy()

#BACKTESTING
serie_backtesting, comision_backtesting, diferencias = backtesting(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, y_hat)
serie_backtesting = pd.DataFrame(serie_backtesting)
serie_backtesting.set_index(activos_close.index[42:])
df_diferencias = pd.concat(diferencias, axis=1).transpose()
plt.figure()
plt.plot(serie_backtesting, label = 'Backtesting Modelo')
plt.title('Backtesting Modelo')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
print(serie_backtesting.iloc[-1])
print(comision_backtesting)
plt.figure()
plt.plot(y_hat)
plt.title('Predicciones')
plt.xlabel('meses')
plt.ylabel('predicciones')


#BACKTESTING REGRESION LOGISTICA
serie_regresion, comision_regresion, diferencias_regresion = backtesting(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, y_hat_regresion)
serie_regresion = pd.DataFrame(serie_regresion)
serie_regresion.set_index(activos_close.index[42:])
df_diferencias_regresion = pd.concat(diferencias_regresion, axis=1).transpose()
plt.figure()
plt.plot(serie_regresion, label = 'Backtesting Modelo "Regresion Logistica"')
plt.title('Backtesting Regresion')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
print(serie_regresion.iloc[-1])
print(comision_regresion)
plt.figure()
plt.plot(y_hat_regresion)
plt.title('Predicciones')
plt.xlabel('epoch')
plt.ylabel('predicciones')



#BACKTESTING RED NEURONAL
serie_red, comision_red, diferencias_red = backtesting(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, y_hat_redneuronal)
serie_red = pd.DataFrame(serie_red)
serie_red.set_index(activos_close.index[42:])
df_diferencias_red = pd.concat(diferencias_red, axis=1).transpose()
plt.figure()
plt.plot(serie_red, label = 'Backtesting Modelo "Red Neuronal"')
plt.title('Backtesting Red Neuronal')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()

print(serie_red.iloc[-1])
print(comision_red)
plt.figure()
plt.plot(y_hat_redneuronal)
plt.title('Predicciones')
plt.xlabel('epoch')
plt.ylabel('predicciones')



# BENCHMARK SINTETICO - EW de las estrategias
serie_bench, comision_benchmark, diferencias = benchmark(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open)
serie_benchmark = pd.DataFrame(serie_bench)
serie_benchmark.set_index(activos_close.index[42:])

plt.figure()
plt.plot(serie_benchmark)
plt.title('Backtesting Benchmark')
plt.xlabel('Fecha')
plt.ylabel('Precio')

serie_benchmark.iloc[-1]


# MONOS - Prueba de aleatoriedad
serie_monos = []
comisiones_monos = []
for i in range(5):
    serie_monos_int, comision_monos, diferencias = monos(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open)
    serie_monos.append(serie_monos_int)
    comisiones_monos.append(comision_monos)
serie_monos = pd.DataFrame(serie_monos).transpose()
serie_monos.set_index(activos_close.index[42:])
comisiones_monos = pd.DataFrame(comisiones_monos).transpose()
plt.figure()
plt.plot(serie_monos)
plt.title('Backtesting Prueba Aleatoriedad')
plt.xlabel('Fecha')
plt.ylabel('Precio')


ranking_monos = serie_monos.iloc[-1,:].sort_values(ascending = False)
fila_central = (ranking_monos.shape[0]-1)//2
mono_50 = ranking_monos[ranking_monos.index[fila_central]]
comision_50 = comisiones_monos.iloc[:,ranking_monos.index[fila_central]]
mono_100 = ranking_monos[ranking_monos.index[0]]
comision_100 = comisiones_monos.iloc[:,ranking_monos.index[0]]

# Se grafican las series de precios de los 3 modelos, el benchmark y el mono del 50%
plt.figure()
plt.plot(serie_benchmark, label = "benchmark sintetico")
plt.plot(serie_red, label = "Red Nueronal")
plt.plot(serie_regresion, label = "Regresion logistica")
plt.plot(serie_backtesting, label = "Modelo")
plt.plot(serie_monos.iloc[:,ranking_monos.index[fila_central]], label = "Mono 50")
plt.plot(serie_monos.iloc[:,ranking_monos.index[0]], label = "Mono 100")
plt.title('Backtesting')
plt.xlabel('fecha')
plt.ylabel('precio')
plt.legend()

# Se concatenan las comisiones de todo lo anterior
plt.figure()
categorias = ['Comision modelo', 'Comision Red Neuronal', 'Comision Regresion', 'Comision Benchmark', 'Comision mono 50%', 'Comision mono 100%']
valores = [comision_backtesting, comision_red, comision_regresion, comision_benchmark, comision_50.iloc[0], comision_100.iloc[0]]
colores = plt.cm.Set1(np.linspace(0, 1, len(categorias)))
fig, ax = plt.subplots()
ax.bar(categorias, valores, color=colores, width=0.1)
ax.tick_params(axis='x', labelsize=5.5)
plt.show()