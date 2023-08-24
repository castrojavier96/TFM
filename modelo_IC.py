import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def modelo_ic(macro_data, etiquetas):

    datos_output = np.asarray(etiquetas).astype('float32') # pasamos los datos float

    datos_output_cat = to_categorical(datos_output, num_classes=4) # se pasa a categoricos como one hot
    print(datos_output_cat.sum(axis=0))

    x_train, x_test, t_train, t_test = train_test_split(macro_data, datos_output_cat, test_size=0.3, random_state=35) # se separan los datos en train y test

    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    model = keras.Sequential()

    #-------------------------------------------------------------
    # TO-DO: Incluye aquí las capas necesarias
    l1_reg = 0.005
    l2_reg = 0.01
    dropout = 0.25

    # Modelo de 3 capas densas, con una salida de 4 y función de activación softmax para que sumen 1
    model.add(Dense(128, input_shape=(29,), activation='relu',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(rate=dropout))

    model.add(Dense(64, input_shape=(128,), activation='relu',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))

    #model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(rate=dropout))



    model.add(Dense(4, input_shape=(64,), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #-------------------------------------------------------------

    model.summary()


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    # Se entrena el modelo
    history = model.fit(x_train, t_train, epochs=500, shuffle=True, validation_split=0.2, callbacks=[early_stop])


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

    macro_data_scaler=scaler.transform(macro_data)
    y_hat = model.predict(macro_data_scaler)
    y_hat = pd.DataFrame(y_hat).idxmax(axis=1)
    y_hat = pd.DataFrame(y_hat, columns=['Valor'])

    # Establecer un nuevo índice para el DataFrame
    new_index = etiquetas.index

    y_hat = y_hat.set_index(pd.Index(new_index))
    y_hat = y_hat.loc['2015-08-24':'2022-12-30']


    return y_hat