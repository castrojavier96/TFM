# ESTRATEGIA MOMENTUM

import numpy as np
import pandas as pd

def benchmark(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open):
    serie_momentum = []
    serie_EW = []
    serie_bench = []
    serie_vol = []
    serie_volmin = []
    diferencias = []
    comision_total = 0
    comision_total_EW = 0
    comision_total_vol = 0
    df_acciones_mes_pasado = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_EW = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_vol = pd.DataFrame(0, index=[0], columns=range(10))
    df_acciones_mes_pasado_volmin = pd.DataFrame(0, index=[0], columns=range(10))  

    for i in range(22,len(activos_close)):
        if activos_close.index[i].month != activos_close.index[i-1].month:
            reb_bench = pd.DataFrame({'A':[1/4], 'B':[1/4], 'C':[1/4], 'D':[1/4]}) #pd.DataFrame(y_hat[0,:]).transpose()
            if i==22:
                capital_momentum = capital_inicial
                capital_EW = capital_inicial
                capital_vol = capital_inicial
                capital_volmin = capital_inicial
            else:
                capital_int = ((n_acciones*activos_close[activos_momentum].iloc[i-1]).sum() + delta) + ((n_acciones_EW*activos_close[activos_EW].iloc[i-1]).sum() + delta_EW) +((n_acciones_vol*activos_close[activos_vol].iloc[i]).sum() + delta_vol)+ ((n_acciones_volmin*activos_close[activos_volmin].iloc[i]).sum() + delta_volmin)

                #capital_momentum_int = reb_monos.iloc[0,0] * ((n_acciones*activos_close[activos_momentum].iloc[i-1]).sum() + delta)
                #capital_EW_int = reb_monos.iloc[0,1] * ((n_acciones_EW*activos_close[activos_EW].iloc[i-1]).sum() + delta_EW)
                capital_momentum = capital_int * reb_bench.iloc[0,0]
                capital_EW = capital_int * reb_bench.iloc[0,1]
                capital_vol = capital_int * reb_bench.iloc[0,2]
                capital_volmin = capital_int * reb_bench.iloc[0,3]

            # MOMENTUM 
            rent_activos = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_momentum = rent_activos.sort_values(ascending = False)[0:n_activos].index
            n_acciones = (capital_momentum/10)//activos_close[activos_momentum].iloc[i-1]#suma producto con el 10% para obtener serie de precios
            df_acciones = pd.DataFrame(n_acciones).transpose()
            diferencia= df_acciones.subtract(df_acciones_mes_pasado, fill_value=0).sum()
            #comision_compra = (activos_close[diferencia[diferencia>0].index].iloc[i]*diferencia[diferencia>0]).sum()*com
            #comision_venta = (activos_open[diferencia[diferencia<0].index].iloc[i]*diferencia[diferencia<0]).sum()*com*-1
            capital_invertido = (n_acciones*activos_close[activos_momentum].iloc[i]).sum()
            delta = capital_momentum - capital_invertido
            #comision_total = comision_total + comision_venta + comision_compra
            df_acciones_mes_pasado = df_acciones

            # EW
            volu_activos = activos_volumen[i-20:i-1].sum()
            activos_EW = volu_activos.sort_values(ascending = False)[0:n_activos].index
            n_acciones_EW = (capital_EW/10)//activos_close[activos_EW].iloc[i-1]#suma producto con el 10% para obtener serie de precios
            df_acciones_EW = pd.DataFrame(n_acciones_EW).transpose()
            diferencia_EW= df_acciones_EW.subtract(df_acciones_mes_pasado_EW, fill_value=0).sum()
            #comision_compra_EW = (activos_close[diferencia_EW[diferencia_EW>0].index].iloc[i]*diferencia_EW[diferencia_EW>0]).sum()*com
            #comision_venta_EW = (activos_open[diferencia_EW[diferencia_EW<0].index].iloc[i]*diferencia_EW[diferencia_EW<0]).sum()*com*-1
            capital_invertido_EW = (n_acciones_EW*activos_close[activos_EW].iloc[i]).sum()
            delta_EW = capital_EW - capital_invertido_EW
            #comision_total_EW = comision_total_EW + comision_compra_EW + comision_venta_EW
            df_acciones_mes_pasado_EW = df_acciones_EW
    
            # Volatilidad
            rent_activos_vol = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_vol = rent_activos_vol.sort_values(ascending = False)[0:n_activos].index
            n_acciones_vol = (capital_vol/10)//activos_close[activos_vol].iloc[i-1]#suma producto con el 10% para obtener serie de precios
            df_acciones_vol = pd.DataFrame(n_acciones_vol).transpose()
            diferencia_vol = df_acciones_vol.subtract(df_acciones_mes_pasado_vol, fill_value=0).sum()
            #comision_compra_vol = (activos_close[diferencia_vol[diferencia_vol>0].index].iloc[i]*diferencia_vol[diferencia_vol>0]).sum()*com
            #comision_venta_vol = (activos_open[diferencia_vol[diferencia_vol<0].index].iloc[i]*diferencia_vol[diferencia_vol<0]).sum()*com*-1
            capital_invertido_vol = (n_acciones_vol*activos_close[activos_vol].iloc[i]).sum()
            delta_vol = capital_vol - capital_invertido_vol
            #comision_total_vol = comision_total_vol + comision_venta_vol + comision_compra_vol
            df_acciones_mes_pasado_vol = df_acciones_vol

            # Volatilidad MINIMA
            rent_activos_volmin = np.log(activos_close).diff()[i-20:i-1].dropna(axis=0).sum(axis=0)
            activos_volmin = rent_activos_volmin.sort_values(ascending = False)[0:n_activos].index
            n_acciones_volmin = (capital_volmin/10)//activos_close[activos_volmin].iloc[i-1]#suma producto con el 10% para obtener serie de precios
            df_acciones_volmin = pd.DataFrame(n_acciones_vol).transpose()
            diferencia_volmin = df_acciones_volmin.subtract(df_acciones_mes_pasado_volmin, fill_value=0).sum()
            #comision_compra_vol = (activos_close[diferencia_vol[diferencia_vol>0].index].iloc[i]*diferencia_vol[diferencia_vol>0]).sum()*com
            #comision_venta_vol = (activos_open[diferencia_vol[diferencia_vol<0].index].iloc[i]*diferencia_vol[diferencia_vol<0]).sum()*com*-1
            capital_invertido_volmin = (n_acciones_volmin*activos_close[activos_volmin].iloc[i]).sum()
            delta_volmin = capital_volmin - capital_invertido_volmin
            #comision_total_vol = comision_total_vol + comision_venta_vol + comision_compra_vol
            df_acciones_mes_pasado_volmin = df_acciones_volmin

            diferencia_total = diferencia.add(diferencia_EW, fill_value=0).add(diferencia_vol, fill_value=0).add(diferencia_volmin, fill_value=0)
            diferencias.append(diferencia_total.loc[diferencia_total != 0])
            comision_compra = (activos_close[diferencia_total[diferencia_total>0].index].iloc[i]*diferencia_total[diferencia_total>0]).sum()*com
            comision_venta = (activos_open[diferencia_total[diferencia_total<0].index].iloc[i]*diferencia_total[diferencia_total<0]).sum()*com*-1
            comision_total = comision_total + comision_venta + comision_compra

        serie_momentum.append((n_acciones*activos_close[activos_momentum].iloc[i]).sum() + delta)
        serie_EW.append((n_acciones_EW*activos_close[activos_EW].iloc[i]).sum() + delta_EW)
        serie_vol.append((n_acciones_vol*activos_close[activos_vol].iloc[i]).sum() + delta_vol)
        serie_volmin.append((n_acciones_volmin*activos_close[activos_volmin].iloc[i]).sum() + delta_volmin)        
        serie_bench.append(((n_acciones*activos_close[activos_momentum].iloc[i]).sum() + delta) + ((n_acciones_EW*activos_close[activos_EW].iloc[i]).sum() + delta_EW) + ((n_acciones_vol*activos_close[activos_vol].iloc[i]).sum() + delta_vol)+ ((n_acciones_volmin*activos_close[activos_volmin].iloc[i]).sum() + delta_volmin))
    return serie_bench, comision_total, diferencias