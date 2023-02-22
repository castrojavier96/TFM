# ESTRATEGIA MOMENTUM

import numpy as np
import pandas as pd

def obtener_precios_mom(activos_close,n_activos,capital_inicial,com):

    # creamos listas a rellenar
    serie_momentum = []
    comision_total = 0 # inicializamos la comision
    for i in range(22,len(activos_close)): # for que va leyendo el dataframe de datos
        if activos_close.index[i].month != activos_close.index[i-1].month: # si hay cambio de mes se rebalancea
            if i==22: # si es el primer dia del dataframe se utiliza el capital inicial, sino la valorizacion de la serie de precios de la estrategia
                capital = capital_inicial
            else:
                capital = delta + (n_acciones*activos_close[activos_momentum].iloc[i-1]).sum()

            rent_activos = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0) # se calcula la rentabilidad de los activos en los ultimos 20 dias
            activos_momentum = rent_activos.sort_values(ascending = False)[0:n_activos].index # se ordenan de mayor a menor y se guardan los 10 primeros
            n_acciones = (capital/10)//activos_close[activos_momentum].iloc[i-1] # se calcula la cantidad de acciones a comprar
            capital_invertido = (n_acciones*activos_close[activos_momentum].iloc[i]).sum() # se calcula cuanto capital queda invertido realmente ya que no se pueden comprar fracciones de una accion
            delta = capital - capital_invertido # calculamos el diferencial entre el capital disponible al que realmente se invirtio
            comision = capital_invertido*com # se calcula la comision
            comision_total = comision_total + comision # se agrega esta comision al costo de comision total acumulada de la estrategia
            
        serie_momentum.append((n_acciones*activos_close[activos_momentum].iloc[i]).sum() + delta) # se obtiene el valor de hoy de la estrategia y se une a la lista para crear la serie de precios
        
    # Aqui se grafica la serie historica de la estrategia junto con la volatilidad de la misma
    serie_momentum = pd.DataFrame(serie_momentum) # se pasa a un Dataframe
    serie_momentum.set_index(activos_close.index[22:], inplace=True) # se ponen las fechas
    serie_momentum.plot() # se grafica

    rent_momentum = np.log(serie_momentum).diff().dropna(axis=0)
    vol_momentum = rent_momentum.rolling(window=20).std() # se calcula la volatilidad de los ultimos 20 dias
    vol_momentum.plot()
    
    return serie_momentum