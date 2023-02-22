# REBALANCEADOR
import subprocess

# Definir el comando que se ejecutará en la terminal
command = 'pip install -r requirements.txt'

# Ejecutar el comando en la terminal desde Python
subprocess.call(command, shell=True)


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

# Incluye aquí otros imports que necesites
#import tensorflow as tf
#import keras
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import Dropout
#from keras.utils.np_utils import to_categorical  
#import matplotlib.pyplot as plt

activos_close, activos_volumen, activos_open = obtener_datos()
n_activos = 10
capital_inicial = 10000
com = 0.0003

serie_momentum = obtener_precios_mom(activos_close,n_activos,capital_inicial,com)
#serie_HRP = obtener_precios_HRP(activos_close)
serie_EW = obtener_precios_EW(activos_close,activos_volumen,n_activos,capital_inicial,com)
serie_volatilidad = obtener_precios_vol(activos_close,n_activos,capital_inicial,com)
serie_volmin  = obtener_precios_volmin(activos_close,n_activos,capital_inicial,com)
inflacion, pib = obtener_macros()

# MODELO
model, datos_inputs = modelo(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, inflacion, pib)
y_hat = model.predict(datos_inputs.values)

#Backtesting
serie_backtesting, comision_backtesting, diferencias = backtesting(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, y_hat)
serie_backtesting = pd.DataFrame(serie_backtesting)
df_diferencias = pd.concat(diferencias, axis=1).transpose()
serie_backtesting.plot()
print(serie_backtesting.iloc[-1])
print(comision_backtesting)


# BENCHMARK SINTETICO - EW de las estrategias
serie_bench, comision_benchmark, diferencias = benchmark(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open)
serie_benchmark = pd.DataFrame(serie_bench)
#serie_monos.set_index(activos_close.index[22:], inplace=True)
serie_benchmark.plot()
serie_benchmark.iloc[-1]


# MONOS - Prueba de aleatoriedad
serie_monos = []
comisiones_monos = []
for i in range(10):
    serie_monos_int, comision_monos, diferencias = monos(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open)
    serie_monos.append(serie_monos_int)
    comisiones_monos.append(comision_monos)
serie_monos = pd.DataFrame(serie_monos).transpose()
comisiones_monos = pd.DataFrame(comisiones_monos).transpose()
#serie_monos.set_index(activos_close.index[22:], inplace=True)
serie_monos.plot()
ranking_monos = serie_monos.iloc[-1,:].sort_values(ascending = False)
fila_central = (ranking_monos.shape[0]-1)//2
mono_50 = ranking_monos.loc[fila_central]
