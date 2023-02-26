# PRUEBA DE ALEATORIEDAD (MONOS)
#Script que recibe como input el dataframede datos y las predicciones hechas por los modelos y simula los monos aleatorios


#Se importan las librerias
import numpy as np
import pandas as pd

#Se define la funcion backtesting con sus inputs
def monos(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, dias_reb):

    # Se inicializan las listas y variable que se van a rellenar
    serie_reb = []
    diferencias = []
    comision_total = 0

    # Dataframes de 0
    df_acciones_mes_pasado = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_EW = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_vol = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_volmin = pd.DataFrame(0, index=[0], columns=range(10))    
    for i in range(42,len(activos_close)):
        if i==42 or (i-1) % dias_reb == 0:#activos_close.index[i].month != activos_close.index[i-1].month:
            reb_monos = pd.DataFrame(np.random.dirichlet(np.ones(5), size=1)) # se guardan las predicciones hechas por los modelos en un dataframe
            if i==42: # condicion de si se iniciando el dataframe para definir el capital a utilizar
                capital_momentum = capital_inicial
                capital_EW = capital_inicial
                capital_vol = capital_inicial
                capital_volmin = capital_inicial
                capital_cash = (capital_inicial*4*100/90)*0.1
            else: # se actualiza de esta forma para los siguientes rebalanceos y se rebalancea segun el input de la prediccion
                capital_int = ((n_acciones * activos_close[activos_momentum].iloc[i - 1]).sum() + delta) + (
                            (n_acciones_EW * activos_close[activos_EW].iloc[i - 1]).sum() + delta_EW) + (
                                          (n_acciones_vol * activos_close[activos_vol].iloc[i]).sum() + delta_vol) + (
                                          (n_acciones_volmin * activos_close[activos_volmin].iloc[i]).sum() + delta_volmin) + capital_cash

                capital_momentum = capital_int * reb_monos.iloc[0,0]
                capital_EW = capital_int * reb_monos.iloc[0,1]
                capital_vol = capital_int * reb_monos.iloc[0,2]
                capital_volmin = capital_int * reb_monos.iloc[0,3]
                capital_cash = capital_int * reb_monos.iloc[0,4]

            #segun cada estrategia se calcula:
            # - Numero de acciones a comprar
            # - Diferencia entre lo que ya habia comprado y lo que se quiere comprar (o vender)
            # - capital que finalmente se invirtio
            # - se actualiza el df de activos comprados
            # MOMENTUM 
            rent_activos = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_momentum = rent_activos.sort_values(ascending = False)[0:n_activos].index
            n_acciones = (capital_momentum/10)//activos_close[activos_momentum].iloc[i-1]#suma producto con el 10% para obtener serie de precios
            df_acciones = pd.DataFrame(n_acciones).transpose()
            diferencia= df_acciones.subtract(df_acciones_mes_pasado, fill_value=0).sum()
            capital_invertido = (n_acciones*activos_close[activos_momentum].iloc[i]).sum()
            delta = capital_momentum - capital_invertido
            df_acciones_mes_pasado = df_acciones

            # EW
            volu_activos = activos_volumen[i-20:i-1].sum()
            activos_EW = volu_activos.sort_values(ascending = False)[0:n_activos].index
            n_acciones_EW = (capital_EW/10)//activos_close[activos_EW].iloc[i-1]
            df_acciones_EW = pd.DataFrame(n_acciones_EW).transpose()
            diferencia_EW= df_acciones_EW.subtract(df_acciones_mes_pasado_EW, fill_value=0).sum()
            capital_invertido_EW = (n_acciones_EW*activos_close[activos_EW].iloc[i]).sum()
            delta_EW = capital_EW - capital_invertido_EW
            df_acciones_mes_pasado_EW = df_acciones_EW

            # Volatilidad
            rent_activos_vol = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_vol = rent_activos_vol.sort_values(ascending = False)[0:n_activos].index
            n_acciones_vol = (capital_vol/10)//activos_close[activos_vol].iloc[i-1]
            df_acciones_vol = pd.DataFrame(n_acciones_vol).transpose()
            diferencia_vol = df_acciones_vol.subtract(df_acciones_mes_pasado_vol, fill_value=0).sum()
            capital_invertido_vol = (n_acciones_vol*activos_close[activos_vol].iloc[i]).sum()
            delta_vol = capital_vol - capital_invertido_vol
            df_acciones_mes_pasado_vol = df_acciones_vol

            # Volatilidad MINIMA
            rent_activos_volmin = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_volmin = rent_activos_volmin.sort_values(ascending = False)[0:n_activos].index
            n_acciones_volmin = (capital_volmin/10)//activos_close[activos_volmin].iloc[i-1]
            df_acciones_volmin = pd.DataFrame(n_acciones_vol).transpose()
            diferencia_volmin = df_acciones_volmin.subtract(df_acciones_mes_pasado_volmin, fill_value=0).sum()
            capital_invertido_volmin = (n_acciones_volmin*activos_close[activos_volmin].iloc[i]).sum()
            delta_volmin = capital_volmin - capital_invertido_volmin
            df_acciones_mes_pasado_volmin = df_acciones_volmin

            # por cada rebalanceo se juntan todas las ordenes de compra/venta de cada estrategia y se calculan las diferencias, ya que probablemente se repitan activos
            # Se calcula las comisiones y se suman al total de comisiones
            diferencia_total = diferencia.add(diferencia_EW, fill_value=0).add(diferencia_vol, fill_value=0).add(diferencia_volmin, fill_value=0)
            diferencias.append(diferencia_total.loc[diferencia_total != 0])
            comision_compra = (activos_close[diferencia_total[diferencia_total>0].index].iloc[i]*diferencia_total[diferencia_total>0]).sum()*com
            comision_venta = (activos_open[diferencia_total[diferencia_total<0].index].iloc[i]*diferencia_total[diferencia_total<0]).sum()*com*-1
            comision_total = comision_total + comision_venta + comision_compra

        # Se guarda el valor del protafolio valorizado todos los dias al close
        serie_reb.append(
            (
                (n_acciones * activos_close[activos_momentum].iloc[i]).sum() + delta +
                (n_acciones_EW * activos_close[activos_EW].iloc[i]).sum() + delta_EW +
                (n_acciones_vol * activos_close[activos_vol].iloc[i]).sum() + delta_vol +
                (n_acciones_volmin * activos_close[activos_volmin].iloc[i]).sum() + delta_volmin +
                capital_cash
            )
        )    
    return serie_reb, comision_total, diferencias # se devuelve como output la serie de precios, la comision de esta y el dataframe con todas las ordenes de compra/venta
