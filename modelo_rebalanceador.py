# REBALANCEADOR

import numpy as np
import pandas as pd
import keras
#from estrategia_1_Momentum import obtener_precios_mom
#from estrategia_2_HRP import obtener_precios_HRP
#from estrategia_3_EW import obtener_precios_EW
#from estrategia_4_Volatilidad import obtener_precios_vol
#from benchmark import benchmark
#from monos import monos
#from datos_s3 import obtener_datos, obtener_inflacion

# Incluye aquí otros imports que necesites
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt

#activos_close, activos_volumen, activos_open = obtener_datos()
#n_activos = 10
#capital_inicial = 10000
#com = 0.0003

#serie_momentum = obtener_precios_mom(activos_close,n_activos,capital_inicial,com)
#serie_HRP = obtener_precios_HRP(activos_close)
#serie_EW = obtener_precios_EW(activos_close,activos_volumen,n_activos,capital_inicial,com)
#serie_volatilidad = obtener_precios_vol(activos_close,n_activos,capital_inicial,com)
#inflacion = obtener_inflacion()

def modelo(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, inflacion, pib):
    serie_momentum.columns = ['Momentum']
    #serie_HRP.columns = ['HRP']
    serie_EW.columns = ['EW']
    serie_volatilidad.columns = ['Vol']
    serie_volmin.columns = ['Vol_min']

    estrategias_close = pd.concat([serie_momentum, serie_EW, serie_volatilidad, serie_volmin], axis=1)
    estrategias_close.plot()


    rent_estrategias = np.log(estrategias_close).diff().dropna(axis=0)#.sum(axis=0) # se calculan la rentabilidad de los activos 

    rent_estrategias_mensual = rent_estrategias.resample('M').sum()
    vol_estrategias_mensual = rent_estrategias.resample('M').std()

    rent_estrategias_mensual.columns = ['Rent Momentum', 'Rent EW', 'Rent volatilidad', 'Rent Volmin']
    vol_estrategias_mensual.columns = ['Vol Momentum', 'Vol EW', 'Vol volatilidad', 'Vol Volmin']

    datos_inputs = pd.concat([rent_estrategias_mensual, vol_estrategias_mensual], axis=1)
    datos_inputs.drop(datos_inputs.index[0], inplace=True)
    datos_inputs.drop(datos_inputs.index[-1], inplace=True)
    datos_inputs.reset_index(drop=True, inplace=True)
    
    inflacion.reset_index(drop=True, inplace=True)
    datos_inputs = pd.concat([datos_inputs, inflacion, pib], ignore_index=True, axis=1)

    datos_output = rent_estrategias_mensual.idxmax(axis=1)

    datos_output.loc[datos_output=='Rent Momentum'] = 0
    datos_output.loc[datos_output=='Rent EW'] = 1
    datos_output.loc[datos_output=='Rent volatilidad'] = 2
    datos_output.loc[datos_output=='Rent Volmin'] = 3

    datos_output.drop(datos_output.index[0:2], inplace=True)

    model = keras.Sequential()

    #-------------------------------------------------------------
    # TO-DO: Incluye aquí las capas necesarias
    l1_reg = 0.005
    l2_reg = 0.01
    dropout = 0.25

    model.add(Dense(128,input_shape=(12,), activation='relu',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(64,input_shape=(128,), activation='relu',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(4,input_shape=(64,), activation='softmax',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #-------------------------------------------------------------

    model.summary()

    datos_output = np.asarray(datos_output).astype('float32')
    datos_output = to_categorical(datos_output, num_classes=4)

    x_train, x_test, t_train, t_test = train_test_split(datos_inputs, datos_output, test_size=0.3, random_state=35)

    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])


    history = model.fit(x_train, t_train, epochs=1000, shuffle=False, validation_split=0.1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Error modelo')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.legend()

    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label = "entrenamiento")
    plt.plot(history.history['val_categorical_accuracy'], label = "validación")
    plt.title('Accuracy Modelo')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    score = model.evaluate(x_test, t_test)
    y_hat = model.predict(x_test)

    plt.figure()
    plt.plot(y_hat, label='predicciones')
    plt.title('Predicciones')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')
    plt.legend()


    y_hat = model.predict(datos_inputs.values)

    plt.figure()
    plt.plot(y_hat, label='predicciones')
    plt.title('Predicciones')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')
    plt.legend()

    return model, datos_inputs