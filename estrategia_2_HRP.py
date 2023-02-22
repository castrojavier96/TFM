# ESTRATEGIA HRP

import numpy as np
import pandas as pd
from datos_s3 import obtener_datos
import riskfolio.HCPortfolio as hc

activos_close, activos_volumen = obtener_datos()
n_activos = 10
capital_inicial = 10000
com = 0.0003

def obtener_precios_HRP(activos_close):
    activos_close.index = pd.to_datetime(activos_close.index)

    def HRP(rentabilidad):
        port = hc(returns=rentabilidad)


        model='HRP' 
        correlation = 'pearson' 
        rm = 'MV' 
        rf = 0 
        linkage = 'single' 
        max_k = 10 
        leaf_order = True 

        w = port.optimization(model=model,
                            correlation=correlation,
                            rm=rm,
                            rf=rf,
                            linkage=linkage,
                            max_k=max_k,
                            leaf_order=leaf_order)
        return w


    serie_HRP = []
    comision_total = 0
    for i in range(22,len(activos_close)):
        if activos_close.index[i].month != activos_close.index[i-1].month:
            if i==22:
                capital = capital_inicial
            else:
                capital = delta + (n_acciones*activos_close[activos_HRP].iloc[i-1]).sum() #(n_acciones*activos_close[activos_momentum].iloc[i-1]).sum()

            rent_activos = np.log(activos_close).diff()[i-20:i].dropna(axis=0)
            rent_activos = pd.DataFrame(rent_activos)
            w = HRP(rent_activos)
            activos_HRP = w.sort_values(ascending = False, by= 'weights')[0:n_activos].index
            n_acciones = (w.sort_values(ascending = False, by= 'weights')[0:n_activos]['weights']* capital)//activos_close[activos_HRP].iloc[i]#suma producto con el 10% para obtener serie de precios
            capital_invertido = (n_acciones*activos_close[activos_HRP].iloc[i]).sum()
            delta = capital - capital_invertido
            comision = capital_invertido*com
            comision_total = comision_total + comision
        serie_HRP.append((n_acciones*activos_close[activos_HRP].iloc[i]).sum() + delta)

    serie_HRP = pd.DataFrame(serie_HRP)
    serie_HRP.set_index(activos_close.index[22:], inplace=True)
    serie_HRP.plot()
    rent_HRP = np.log(serie_HRP).diff().dropna(axis=0)
    vol_HRP = rent_HRP.rolling(window=20).std()
    vol_HRP.plot()

    return serie_HRP