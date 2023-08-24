import pandas as pd
import numpy as np
import glob
import os
import pandas_datareader.data as wb
import pandas as pd
from datetime import datetime
from fredapi import Fred
import keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sn

def etiquetasic():
   def InvestmentClockFixed(signals,cuts={'GDP':2.5,'CPI':2}):
      
      signals = signals.copy().assign(Growth=None,Inflation=None,Theme=None)
      
      # define high and low growth
      signals.loc[signals['GDP']<=cuts['GDP'],'Growth'] = 'low'
      signals.loc[signals['GDP']>cuts['GDP'],'Growth'] = 'high'
      
      # define high and low inflation  
      signals.loc[signals['CPI']<=cuts['CPI'],'Inflation'] = 'low'
      signals.loc[signals['CPI']>cuts['CPI'],'Inflation'] = 'high'
      
      # define investment clock phases
      signals.loc[(signals.Growth=='low')&(signals.Inflation=='low'),'Theme'] = 'Reflection'
      signals.loc[(signals.Growth=='high')&(signals.Inflation=='low'),'Theme'] = 'Recovery'
      signals.loc[(signals.Growth=='high')&(signals.Inflation=='high'),'Theme'] = 'Overheat'
      signals.loc[(signals.Growth=='low')&(signals.Inflation=='high'),'Theme'] = 'Stagflation'
      return signals.dropna()

   import pandas_datareader.data as wb

   tickersfred = ['GDPC1', 'CPILFESL']
   data = wb.DataReader(tickersfred, 'fred', '2010-1-1') 

   pruebas = data.rename(columns={'GDPC1': 'GDP',
   'CPILFESL': 'CPI'
   })
   pruebas=pruebas.pct_change(12).dropna()*100
   new_index = pd.date_range(start=pruebas.index.min(), end=pruebas.index.max(), freq='B')
   df_daily = pruebas.reindex(new_index, method='ffill')

   prueba = InvestmentClockFixed(df_daily)

   etiquetas = prueba.loc['2013-08-19':'2023-08-18',['Theme']]
   word_to_number = {'Recovery': 0, 'Reflection': 1, 'Overheat': 2, 'Stagflation': 3}
   etiquetas = etiquetas.replace(word_to_number)


   etiquetas = etiquetas.loc['2013-08-19':'2022-12-30']

   return etiquetas
