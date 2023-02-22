# MODELO REBALANCEADOR

#Importamos las librerias
import numpy as np
import pandas as pd
import keras

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

#Se define la funcion para poder crear y entrenar el modelo
def modelo(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, inflacion, pib): # como inputs recibe las series de precios de las estrategias y datos macroeconomicos
    #Le ponemos nombres a las columnas
    serie_momentum.columns = ['Momentum']
    #serie_HRP.columns = ['HRP']
    serie_EW.columns = ['EW']
    serie_volatilidad.columns = ['Vol']
    serie_volmin.columns = ['Vol_min']

    # concatenamos las series de precio en un solo dataframe
    estrategias_close = pd.concat([serie_momentum, serie_EW, serie_volatilidad, serie_volmin], axis=1)
    estrategias_close.plot() # se grafican para ver que pinta tienen

    # Se calcula la rentabilidad de diaria de los datos
    rent_estrategias = np.log(estrategias_close).diff().dropna(axis=0)#.sum(axis=0) # se calculan la rentabilidad de los activos 
    # Se hace un resample para tener datos de rentabilidad y volatilidad mensual
    rent_estrategias_mensual = rent_estrategias.resample('M').sum()
    vol_estrategias_mensual = rent_estrategias.resample('M').std()

    # Se les pone nombre a las columnas de los nuevos dataframes
    rent_estrategias_mensual.columns = ['Rent Momentum', 'Rent EW', 'Rent volatilidad', 'Rent Volmin']
    vol_estrategias_mensual.columns = ['Vol Momentum', 'Vol EW', 'Vol volatilidad', 'Vol Volmin']

    # concatenamos todo y se dropean filas con NA
    datos_inputs = pd.concat([rent_estrategias_mensual, vol_estrategias_mensual], axis=1)
    datos_inputs.drop(datos_inputs.index[0], inplace=True)
    datos_inputs.drop(datos_inputs.index[-1], inplace=True)
    datos_inputs.reset_index(drop=True, inplace=True)
    
    # Se agregan los datos Macroeconomicos como la inflacion y el PIB
    inflacion.reset_index(drop=True, inplace=True)
    datos_inputs = pd.concat([datos_inputs, inflacion, pib], ignore_index=True, axis=1) 

    # para el output del modelo se obtiene que estrategia le fue mejor en el mes siguiente
    datos_output = rent_estrategias_mensual.idxmax(axis=1)
    # Se les asigna un numero, para poder utilizarlo como output
    datos_output.loc[datos_output=='Rent Momentum'] = 0
    datos_output.loc[datos_output=='Rent EW'] = 1
    datos_output.loc[datos_output=='Rent volatilidad'] = 2
    datos_output.loc[datos_output=='Rent Volmin'] = 3
    #Se dropean filas con NA
    datos_output.drop(datos_output.index[0:2], inplace=True)

    datos_output = np.asarray(datos_output).astype('float32') # pasamos los datos float
    datos_output_cat = to_categorical(datos_output, num_classes=4) # se pasa a categoricos como one hot
    print(datos_output_cat.sum(axis=0))
    
    x_train, x_test, t_train, t_test = train_test_split(datos_inputs, datos_output_cat, test_size=0.3, random_state=35) # se separan los datos en train y test
    # Se estandarizan los datos
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    #Se define el modelo
    model = keras.Sequential()

    #-------------------------------------------------------------
    # TO-DO: Incluye aquí las capas necesarias
    l1_reg = 0.005
    l2_reg = 0.01
    dropout = 0.25
    # Modelo de 3 capas densas, con una salida de 4 y funcion de activacion softmax para que sumen 1
    model.add(Dense(128,input_shape=(12,), activation='relu',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(64,input_shape=(128,), activation='relu',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(4,input_shape=(64,), activation='softmax',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #-------------------------------------------------------------

    model.summary()
    # Optimizador adam, loss categorical crossentropy y early Stopping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    # Se entrena el modelo
    history = model.fit(x_train, t_train, epochs=1000, shuffle=True, validation_split=0.1, callbacks=[early_stop])
    # Se grafica el error
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Error modelo')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.legend()
    # Se grafica el accuracy
    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label = "entrenamiento")
    plt.plot(history.history['val_categorical_accuracy'], label = "validación")
    plt.title('Accuracy Modelo')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    score = model.evaluate(x_test, t_test) # Evaluamos el modelo
    y_hat = model.predict(x_test) # Se hacen las predicciones de test
    # Se grafican estas predicciones
    plt.figure()
    plt.plot(y_hat, label='predicciones')
    plt.title('Predicciones')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')
    plt.legend()
    # Se hacen predicciones para todo el Dataframe y se grafican
    y_hat = model.predict(datos_inputs.values)

    plt.figure()
    plt.plot(y_hat, label='predicciones')
    plt.title('Predicciones')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')
    plt.legend()

    return model, datos_inputs, datos_output # nos quedamos con el modelo y los datos inputs