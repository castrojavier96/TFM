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
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sn

#Se define la funcion para poder crear y entrenar el modelo
def modelo_reb(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, etiquetas): # como inputs recibe las series de precios de las estrategias y datos macroeconomicos
    #Le ponemos nombres a las columnas
    serie_momentum.columns = ['Momentum']
    serie_EW.columns = ['EW']
    serie_volatilidad.columns = ['Vol']
    serie_volmin.columns = ['Vol_min']
    #serie_cash = pd.DataFrame(pd.Series([10000] * len(serie_momentum))).set_index(serie_momentum.index)

    # concatenamos las series de precio en un solo dataframe
    estrategias_close = pd.concat([serie_momentum, serie_EW, serie_volatilidad, serie_volmin], axis=1)
    #estrategias_close = estrategias_close.rename(columns={0:'Cash'})
    #estrategias_close = estrategias_close.iloc[42:,:]

    returns = estrategias_close.pct_change()
    mean_daily_returns = returns.rolling(window=30).mean()
    std_dev_daily_returns = returns.rolling(window=30).std()
    rolling_sharpe_ratio = (mean_daily_returns/ std_dev_daily_returns) * np.sqrt(30)
    rolling_sharpe_ratio.dropna(axis=0, inplace=True)


    mean_daily_returns_15 = returns.rolling(window=15).mean()
    std_dev_daily_returns_15 = returns.rolling(window=15).std()
    rolling_sharpe_ratio_15 = (mean_daily_returns_15/ std_dev_daily_returns_15) * np.sqrt(15)
    rolling_sharpe_ratio_15.dropna(axis=0, inplace=True)
    rolling_sharpe_ratio_15 = rolling_sharpe_ratio_15.loc['2015-11-04':]

    mean_daily_returns_1 = returns.rolling(window=1).mean()
    std_dev_daily_returns_1 = returns.rolling(window=1).std()
    rolling_sharpe_ratio_1 = (mean_daily_returns_1/ std_dev_daily_returns_1) * np.sqrt(1)
    rolling_sharpe_ratio_1.dropna(axis=0, inplace=True)
    rolling_sharpe_ratio_1 = rolling_sharpe_ratio_15.loc['2015-11-04':]


    # Se calcula la rentabilidad de diaria de los datos
    rent_estrategias = np.log(estrategias_close).diff().dropna(axis=0)#.sum(axis=0) # se calculan la rentabilidad de los activos 
    rent_estrategias.corr()

    ventana_rodante = rent_estrategias.rolling('30D')
    rentabilidad_ultimos_30_dias = ventana_rodante.apply(lambda x: (1 + x).prod() - 1)
    volatilidad_ultimos_30_dias = ventana_rodante.std()

    # Se hace un resample para tener datos de rentabilidad y volatilidad mensual
    #rent_estrategias_mensual = rent_estrategias.resample('M').sum()
    #vol_estrategias_mensual = rent_estrategias.resample('M').std()

    # Se les pone nombre a las columnas de los nuevos dataframes
    rentabilidad_ultimos_30_dias.columns = ['Rent Momentum', 'Rent EW', 'Rent volatilidad', 'Rent Volmin']
    volatilidad_ultimos_30_dias.columns = ['Vol Momentum', 'Vol EW', 'Vol volatilidad', 'Vol Volmin']

    # concatenamos todo y se dropean filas con NA
    #datos_inputs = pd.concat([rentabilidad_ultimos_30_dias, volatilidad_ultimos_30_dias], axis=1)
    #datos_inputs.drop(datos_inputs.index[0], inplace=True)
    #datos_inputs.drop(datos_inputs.index[-1], inplace=True)
    #datos_inputs.reset_index(drop=True, inplace=True)
    
    
    datos_inputs = pd.concat([rolling_sharpe_ratio, rolling_sharpe_ratio_15, etiquetas], axis=1)
    datos_inputs = datos_inputs.loc['2015-11-04':'2023-06-29']

    # Se agregan los datos Macroeconomicos como la inflacion
    #inflacion.reset_index(drop=True, inplace=True)
    #datos_inputs = pd.concat([datos_inputs, inflacion], ignore_index=True, axis=1) 

    # para el output del modelo se obtiene que estrategia le fue mejor en el mes siguiente
    datos_output = rolling_sharpe_ratio_1.idxmax(axis=1)
    # Se les asigna un numero, para poder utilizarlo como output
    datos_output.loc[datos_output=='Momentum'] = 0
    datos_output.loc[datos_output=='EW'] = 1
    datos_output.loc[datos_output=='Vol'] = 2
    datos_output.loc[datos_output=='Vol_min'] = 3
    #datos_output.loc[datos_output=='Rent cash'] = 4
    #Se dropean filas con NA
    #datos_output.drop(datos_output.index[0:1], inplace=True)
    

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
    
    # Modelo de 3 capas densas, con una salida de 4 y función de activación softmax para que sumen 1
    model.add(Dense(64, input_shape=(9,), activation='relu',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(rate=dropout))
    
    model.add(Dense(32, input_shape=(64,), activation='ELU',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(rate=dropout))
    
    model.add(Dense(4, input_shape=(32,), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #-------------------------------------------------------------
    
    model.summary()

    # Optimizador adam, loss categorical crossentropy y early Stopping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    # Se entrena el modelo
    history = model.fit(x_train, t_train, epochs=500, shuffle=True, validation_split=0.1, callbacks=[early_stop])
    # Se grafica el error
    plt.figure()
    plt.plot(history.history['loss'], label = "entrenamiento")
    plt.plot(history.history['val_loss'], label = "validación")
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
    # Se imprimen el accuracy en train y test
    print("Accuracy (train):", model.evaluate(x_train, t_train))
    print("Accuracy (test):", model.evaluate(x_test, t_test))

    y_hat = model.predict(x_test) # Se hacen las predicciones de test
    # Se grafican estas predicciones
    plt.figure()
    plt.plot(y_hat, label='predicciones')
    plt.title('Predicciones')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')
    plt.legend()
    y_hat[:,:-1].sum(axis=1)
    # Se hacen predicciones para todo el Dataframe y se grafican
    datos_inputs_esc = scaler.transform(datos_inputs.values)
    
    # se calcula la matriz de confucion y se grafica
    matrix = confusion_matrix(np.argmax(t_test,axis = 1), np.argmax(y_hat, axis=1))
    plt.figure()
    sn.heatmap(matrix, annot=True)
    plt.title('Matriz de Confusion Modelo')
    plt.show()

    # Se grafican los falsos positivos y falsos negativos
    plt.figure(figsize=(16, 4))
    for i in range(4):
        fpr, tpr, thresholds = roc_curve(np.argmax(t_test,axis = 1) == i, np.argmax(y_hat, axis=1) == i)
        plt.subplot(1,5,i+1)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.grid()
    plt.show()

    # Se grafica el recall  y la presicion
    plt.figure(figsize=(16, 4))
    for i in range(4):
        prec, recall, _ = precision_recall_curve(np.argmax(t_test,axis = 1) == i, np.argmax(y_hat, axis=1) == i)
        print(recall, prec)
        plt.subplot(1,5,i+1)
        plt.plot(recall, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(-0.05, 1.05)
        plt.grid()
    plt.show()

    return model, datos_inputs, datos_output, datos_inputs_esc # nos quedamos con el modelo y los datos inputs
