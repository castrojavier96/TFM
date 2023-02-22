# MODELO REGRESION LOGISTICA MULTICLASE (VISTO EN CLASES)

# Se importan las librerias
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Se define la funcion
def regresion_logistica(datos_inputs, datos_output):

  # Dividir los datos en conjuntos de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(datos_inputs, datos_output, test_size=0.3, random_state=35)
  # Se estandarizan los datos
  scaler=StandardScaler()
  X_train=scaler.fit_transform(X_train)
  X_test=scaler.transform(X_test)

  # Se define una clase para crear la REGRESION LOGISTICA
  class LogisticRegressionModel:

    def __init__(self, d0=2):
      self.W = tf.Variable(tf.random.normal(shape=[d0, 1], dtype=tf.dtypes.float64))  
      self.b = tf.Variable(tf.random.normal(shape=[1], dtype=tf.dtypes.float64)) 

    def predict(self, x):
      """
      x must be a (n,d0) array
      returns a (n,1) array with the predictions for each of the n patterns
      """
      z = tf.matmul(x, self.W) + self.b
      y = tf.math.sigmoid(z)
      return y

    def loss(self, x, t):
      """
      computes the cross-entropy between the model predictions and the targets
      """
      y = self.predict(x)
      loss = tf.reduce_mean(-t*tf.math.log(y) - (1.-t)*tf.math.log(1.-y), axis=0)
      return loss

    def fit(self, x, t, eta, num_epochs):
      """
      Fits the model parameters with data (x, t) using a learning rate eta and
      num_epochs epochs
      """
      loss_history = []
      for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
          loss = self.loss(x, t)

        loss_history.append(loss.numpy().ravel()[0])
        
        [db, dW] = tape.gradient(loss, [self.b, self.W])
        self.b.assign(self.b - eta*db)
        self.W.assign(self.W - eta*dW)
        
      return loss_history

    def accuracy(self, x, t):
      y = self.predict(x).numpy()
      pred = y > 0.5
      return np.mean(pred == t)

  # Se define una clase para hacerla multiclase
  class MulticlassLogisticRegressionModel:

    def __init__(self, input_dimension, num_classes):
      self.models = [LogisticRegressionModel(input_dimension) for _ in range(num_classes)]
      self.num_classes = num_classes

    def fit(self, x, t, eta, num_epochs):
      for id_class, model in enumerate(self.models):
        t_class = (t == id_class) * 1
        model.fit(x, t_class, eta, num_epochs)

    def predict(self, x):
      # x tiene shape (n, input_dimension)
      # y tiene shape (n, num_classes)
      preds = np.zeros((x.shape[0], self.num_classes))
      for id_class, model in enumerate(self.models):
        preds[:, id_class] = model.predict(x)[:, 0]

      # Normalizacion
      preds /= preds.sum(axis=1, keepdims=True)

      return preds

    def accuracy(self, x, t):
      preds = self.predict(x)
      y = np.argmax(preds, axis=1)[:, None]
      return np.mean(y == t)


  # Construccion del modelo
  n, d = X_train.shape
  num_clases = len(np.unique(y_train)) # definimos el numero de clases
  # Se inicializa el modelo de regresion logistica
  model = MulticlassLogisticRegressionModel(d, num_clases)

  # Entrenamiento del modelo
  eta = 0.8
  epochs = 200

  y_train = y_train.reshape(y_train.shape[0],1)
  model.fit(X_train, y_train, eta, epochs) # Se entrena el modelo
  # Se calcula el accuracy tanto en train como test
  print("Accuracy (train):", model.accuracy(X_train, y_train))
  print("Accuracy (test):", model.accuracy(X_test, y_test))
  
  return model