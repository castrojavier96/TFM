# MODELO RED NEURONAL MULTICLASE (VISTO EN CLASES)

# Se importan las librerias
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sn

# Se define la funcion
def red_neuronal(datos_inputs, datos_output):

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(datos_inputs, datos_output, test_size=0.3, random_state=35)
    # Se estandarizan los datos
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    # Se define una clase para crear la red neuronal
    class NeuralNetworkModel:

        def __init__(self, layers_size=[2]):
            self.W = [tf.Variable(tf.random.normal(shape=[a, b], dtype=tf.dtypes.float64)) for a, b in zip(layers_size[:-1], layers_size[1:])]
            self.b = [tf.Variable(tf.random.normal(shape=[1, b], dtype=tf.dtypes.float64)) for b in layers_size[1:]]

        def predict_pre_activation(self, x):
            """
            x must be a (n,d0) array
            returns a (n,num_clases) array with the pre-activations for each of the n patterns
            """
            y = x
            for w, b in zip(self.W[:-1], self.b[:-1]):
                z = tf.matmul(y, w) + b
                y = tf.nn.sigmoid(z)

            z = tf.matmul(y, self.W[-1]) + self.b[-1]

            return z

        def predict(self, x):
            """
            x must be a (n,d0) array
            returns a (n,num_clases) array with the predictions for each of the n patterns
            """
            y = x
            for w, b in zip(self.W[:-1], self.b[:-1]):
                z = tf.matmul(y, w) + b
                y = tf.nn.sigmoid(z)

            z = tf.matmul(y, self.W[-1]) + self.b[-1]
            y = tf.nn.softmax(z, axis=1)
        
            return y

        def loss(self, x, t):
            """
            computes the MSE between the model predictions and the targets
            """
            z = self.predict_pre_activation(x)
            xentropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(t, z) # Esta función necesita que pasemos los logits
            loss = tf.reduce_mean(xentropy)
            return loss

        def fit(self, x, t, eta, num_epochs):
            """
            Fits the model parameters with data (x, t) using a learning rate eta and
            num_epochs epochs
            """
            loss_history = []
            for epoch in range(num_epochs):
                with tf.GradientTape(persistent=True) as tape:
                    loss = self.loss(x, t)

                loss_history.append(loss.numpy().ravel()[0])
            
                for b, W in zip(self.b, self.W):
                    [db, dW] = tape.gradient(loss, [b, W])
                    b.assign(b - eta*db)
                    W.assign(W - eta*dW)
            
            return loss_history

        def accuracy(self, x, t):
            preds = self.predict(x).numpy()
            y = np.argmax(preds, axis=1)[:, None]
            return np.mean(y == t)

    # Se define el numero de epocas
    nepocas = 500
    eta = 0.1
    # Se inicializa el modelo
    model = NeuralNetworkModel([12, 20, 4])

    y_train = y_train.reshape(y_train.shape[0],1).astype(np.int64)# Se hace el reshape para pasar de (81,) a (81,1)
    y_test = y_test.reshape(y_test.shape[0],1)
    loss = model.fit(X_train, y_train, eta, nepocas) # se entrena el modelo
    # se grafica el error
    plt.plot(loss)
    plt.grid(True)
    plt.xlabel("época")
    plt.ylabel("Cross-entropy loss")
    plt.show()
    # se hacen predicciones
    preds_train = model.predict(X_train).numpy()
    preds_test = model.predict(X_test).numpy()
    # Se calcula el accuracy tanto en train como test
    print("Accuracy (train):", model.accuracy(X_train, y_train))
    print("Accuracy (test):", model.accuracy(X_test, y_test))

    # se calcula la matriz de confucion y se grafica
    matrix = confusion_matrix(y_test[:, 0], np.argmax(preds_test, axis=1))

    sn.heatmap(matrix, annot=True)
    plt.show()

    # Se grafican los falsos positivos y falsos negativos
    plt.figure(figsize=(16, 4))
    for i in range(3):
        fpr, tpr, thresholds = roc_curve(y_test[:, 0] == i, np.argmax(preds_test, axis=1) == i)
        plt.subplot(1,3,i+1)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.grid()
    plt.show()

    # Se grafica el recall  y la presicion
    plt.figure(figsize=(16, 4))
    for i in range(3):
        prec, recall, _ = precision_recall_curve(y_test[:, 0] == i, np.argmax(preds_test, axis=1) == i)
        print(recall, prec)
        plt.subplot(1,3,i+1)
        plt.plot(recall, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(-0.05, 1.05)
        plt.grid()
    plt.show()


    return model # como output de la funcion se devuelve el modelo de red neuronal