# ESTRATEGIA Volatilidad max

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def obtener_precios_vol(activos_close,n_activos,capital_inicial,com):
    # creamos listas a rellenar
    serie_Volatilidad = []
    comision_total = 0# inicializamos la comision
    for i in range(22,len(activos_close)):# for que va leyendo el dataframe de datos
        if activos_close.index[i].month != activos_close.index[i-1].month:# si hay cambio de mes se rebalancea
            if i==22: # si es el primer dia del dataframe se utiliza el capital inicial, sino la valorizacion de la serie de precios de la estrategia
                capital = capital_inicial
            else:
                capital = delta + (n_acciones*activos_close[activos_Volatilidad].iloc[i-1]).sum() #(n_acciones*activos_close[activos_Volatilidad].iloc[i-1]).sum()

            rent_activos = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0)# se calcula la rentabilidad de los activos en los ultimos 20 dias
            vol_activos = rent_activos.std() # se calcula la volatilidad de la rentabilidad de los ultimos 20 dias
            activos_Volatilidad = vol_activos.sort_values(ascending = False)[0:n_activos].index# se ordenan de mayor a menor y se guardan los 10 primeros
            n_acciones = (capital/10)//activos_close[activos_Volatilidad].iloc[i-1] # se calcula la cantidad de acciones a comprar
            capital_invertido = (n_acciones*activos_close[activos_Volatilidad].iloc[i]).sum()# se calcula cuanto capital queda invertido realmente ya que no se pueden comprar fracciones de una accion
            delta = capital - capital_invertido# calculamos el diferencial entre el capital disponible al que realmente se invirtio
            comision = capital_invertido*com# se calcula la comision
            comision_total = comision_total + comision# se agrega esta comision al costo de comision total acumulada de la estrategia

        serie_Volatilidad.append((n_acciones*activos_close[activos_Volatilidad].iloc[i]).sum() + delta)# se obtiene el valor de hoy de la estrategia y se une a la lista para crear la serie de precios

    # Aqui se grafica la serie historica de la estrategia junto con la volatilidad de la misma
    serie_Volatilidad = pd.DataFrame(serie_Volatilidad)
    serie_Volatilidad.set_index(activos_close.index[22:], inplace=True)
    # Se grafica la serie historica de la estrategia
    plt.figure()
    plt.plot(serie_Volatilidad, label = 'Serie Volatilidad Maxima')
    plt.title('Serie Precios Volatilidad Max')
    plt.xlabel('Fechas')
    plt.ylabel('Precio')
    plt.legend()


    rent_Vol = np.log(serie_Volatilidad).diff().dropna(axis=0)
    vol_Volatilidad = rent_Vol.rolling(window=20).std()
    # Se grafica la volatilidad historica de la estrategia
    plt.figure()
    plt.plot(vol_Volatilidad, label = 'Volatilidad Serie Volatilidad maxima')
    plt.title('Volatilidad Serie Volatilidad')
    plt.xlabel('Fechas')
    plt.ylabel('Volatilidad')
    plt.legend()   


    return serie_Volatilidad
