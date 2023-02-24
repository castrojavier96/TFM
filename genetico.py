import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from deap import algorithms, base, creator, tools
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

from sklearn.metrics import accuracy_score
from genetic_algorithm import GeneticAlgorithm # asumiendo que ya tienes la implementación del algoritmo genético

def modelo_genetico(datos_inputs,datos_output):
    datos_output_cat = to_categorical(datos_output, num_classes=5)
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
    model.add(Dense(5,input_shape=(64,), activation='softmax',kernel_initializer=keras.initializers.glorot_normal(),kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
    #-------------------------------------------------------------
    # Función de aptitud (fitness function) para el algoritmo genético
    def fitness_function(solution):
        global X_train, y_train, X_test, y_test, model
        model_weights = []
        for i in range(0, len(solution), 2):
            w = np.array(solution[i]).reshape(model.layers[i//2].get_weights()[0].shape)
            b = np.array(solution[i+1]).reshape(model.layers[i//2].get_weights()[1].shape)
            model_weights.append(w)
            model_weights.append(b)
        model.set_weights(model_weights)
        y_pred = model.predict(X_train)
        score = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))
        return score

    # Crear una instancia del algoritmo genético
    ga = GeneticAlgorithm(population_size=20, chromosome_length=785, fitness_function=fitness_function)

    # Ejecutar el algoritmo genético
    best_solution, best_fitness = ga.run(generations=10)

    # Evaluar la mejor solución en el conjunto de prueba
    model_weights = []
    for i in range(0, len(best_solution), 2):
        w = np.array(best_solution[i]).reshape(model.layers[i//2].get_weights()[0].shape)
        b = np.array(best_solution[i+1]).reshape(model.layers[i//2].get_weights()[1].shape)
        model_weights.append(w)
        model_weights.append(b)
    model.set_weights(model_weights)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print('Precisión en el conjunto de prueba:', test_accuracy)
    return model