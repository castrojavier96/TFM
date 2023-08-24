from data_downloader import DataDownloader
from etf_downloader import ETFDataDownloader
from s3_uploader import S3Uploader
import pandas as pd
import numpy as np
import os
from investmentclock import etiquetasic
from modelo_IC import modelo_ic
from estrategia_1_Momentum import obtener_precios_mom
from estrategia_3_EW import obtener_precios_EW
from estrategia_4_Volatilidad import obtener_precios_vol
from estrategia_5_MINVolatilidad import obtener_precios_volmin
from graficos import graficar_series_precio
from modelo_rebalanceador import modelo_reb
from backtesting import backtesting
import matplotlib.pyplot as plt
from benchmark import benchmark
from regresion_logistica import regresion_logistica
from monos import monos
from scipy.stats import percentileofscore




if __name__ == "__main__":

    uploader = S3Uploader()

    n_activos = 10
    capital_inicial = 10000
    com = 0.0003
    dias_reb = 15

    folder_path = 'ETF/'
    dataframes = uploader.download_file(folder_path)

    folder_path = 'macro/'
    dataframes_macro = uploader.download_file_macro(folder_path)
 
    etiquetas = etiquetasic()

    #macro_data= pd.read_csv('macro_data.csv',index_col=0)
    macro_data=dataframes_macro[0]
    macro_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    macro_data.set_index('Date', inplace=True)
    macro_data.index = pd.to_datetime(macro_data.index)
    new_index = pd.date_range(start=macro_data.index.min(), end=macro_data.index.max(), freq='B')
    macro_data = macro_data.reindex(new_index, method='ffill')
    macro_data = macro_data.loc['2013-08-19':'2022-12-30']
    macro_data = macro_data.iloc[:,2:]


    etiquetas_ic = modelo_ic(macro_data, etiquetas)

    #activos_volumen= pd.read_csv('etf_data_volumen.csv',index_col=0)
    activos_volumen= dataframes[2]
    activos_volumen.index = pd.to_datetime(activos_volumen.index)
    new_index = pd.date_range(start=activos_volumen.index.min(), end=activos_volumen.index.max(), freq='B')
    activos_volumen = activos_volumen.reindex(new_index, method='ffill')
    activos_volumen = activos_volumen.loc[:'2022-12-30']
    activos_volumen

    #activos_close= pd.read_csv('etf_data.csv',index_col=0)
    activos_close= dataframes[0]
    activos_close.index = pd.to_datetime(activos_close.index)
    new_index = pd.date_range(start=activos_close.index.min(), end=activos_close.index.max(), freq='B')
    activos_close = activos_close.reindex(new_index, method='ffill')
    activos_close = activos_close.loc['2015-08-24':'2022-12-30']

    #activos_open= pd.read_csv('etf_data_open.csv',index_col=0)
    activos_open= dataframes[1]
    activos_open.index = pd.to_datetime(activos_open.index)
    new_index = pd.date_range(start=activos_open.index.min(), end=activos_open.index.max(), freq='B')
    activos_open = activos_open.reindex(new_index, method='ffill')
    activos_open = activos_open.loc['2015-08-24':'2022-12-30']

    common_columns = list(set(activos_volumen.columns) & set(activos_close.columns) & set(activos_open.columns))
    activos_volumen = activos_volumen[common_columns]
    activos_close = activos_close[common_columns]
    activos_open = activos_open[common_columns]

    # Se calcula la rentabilidad de los activos y se visualizan las rentabilidades
    df_retornos = activos_close.pct_change()
    df_rentabilidad = (1 + df_retornos).cumprod()

    # Obtenemos los 5 activos con mayor rentabilidad acumulada al final del periodo
    activos_top5 = df_rentabilidad.iloc[-1].sort_values(ascending=False)[:5].index.tolist()
    activos_bot5 = df_rentabilidad.iloc[-1].sort_values(ascending=True)[:5].index.tolist()
    activos_top_bot = lista3 = activos_top5 + activos_bot5

    # Graficamos la rentabilidad acumulada de los 5 activos con mayor y menor rentabilidad acumulada
    fig, ax = plt.subplots(figsize=(10, 5))
    df_rentabilidad[activos_top_bot].plot(ax=ax)

    # Creamos la leyenda para los 5 primeros y los 5 últimos activos
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [label for label in labels if label in activos_top5 or label in activos_bot5]
    plt.legend(handles, new_labels, loc='upper left')
    plt.title('Rentabilidad acumulada de los 5 primeros y los 5 últimos activos')
    plt.xlabel('Fecha')
    plt.ylabel('Rentabilidad acumulada')
    plt.grid(True)
    plt.show()


    serie_momentum = obtener_precios_mom(activos_close, n_activos, capital_inicial, com, dias_reb, etiquetas_ic)
    serie_EW = obtener_precios_EW(activos_close, activos_volumen, n_activos, capital_inicial, com, etiquetas_ic)
    serie_volatilidad = obtener_precios_vol(activos_close, n_activos, capital_inicial, com, dias_reb, etiquetas_ic)
    serie_volmin = obtener_precios_volmin(activos_close, n_activos, capital_inicial, com, dias_reb, etiquetas_ic)

    graficar_series_precio(serie_momentum, serie_EW, serie_volatilidad, serie_volmin)

    # MODELO (SE DEFINE Y ENTRENA NUESTRO MODELO, Y NOS QUEDAMOS CON EL OUTPUT Y LAS PREDICCIONES)
    model, datos_inputs, datos_output, datos_inputs_esc = modelo_reb(serie_momentum, serie_EW, serie_volatilidad, serie_volmin, etiquetas_ic)
    y_hat = model.predict(datos_inputs_esc)
    y_hat_binario = np.where(y_hat > 0.5, 1, 0)

    y_hat_reb = pd.DataFrame(y_hat).idxmax(axis=1)

    # BACKTESTING DE NUESTRO MODELO
    serie_backtesting, comision_backtesting, diferencias, ratio_sharpe_modelo, ratio_sortino_modelo, drawdown, maxdradown_modelo = backtesting(activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, y_hat, dias_reb, etiquetas_ic)

    # Calculamos los retornos acumulados y guardamos un Dataframe con todas las ordenes de compra
    df_retornos = serie_backtesting.pct_change()
    serie_backtesting = (1 + df_retornos).cumprod()
    df_diferencias = pd.concat(diferencias, axis=1).transpose()

    # Se grafica la rentabilidad acumulada, las predicciones, el drawdown y los resultados
    plt.figure()
    plt.plot(serie_backtesting, label='Backtesting Modelo')
    plt.title('Backtesting Modelo')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()

    plt.figure()
    plt.plot(y_hat_reb)
    plt.title('Predicciones Modelo')
    plt.xlabel('meses')
    plt.ylabel('predicciones')

    plt.figure()
    plt.plot(drawdown, label='Drawdown Modelo')
    plt.title('Drawdown Modelo')
    plt.xlabel('Fecha')
    plt.ylabel('Porcentaje')
    plt.legend()

    print(f"La rentabilidad acumulada del modelo es: {round(serie_backtesting.iloc[-1]*100, 2)}%")
    print(f"La comision total del modelo es: {comision_backtesting}")
    print(f"El ratio de Sharpe del modelo es: {ratio_sharpe_modelo}")
    print(f"El ratio de Sortino del modelo es: {ratio_sortino_modelo}")
    print(f"El Maximo drawdown del modelo es: {maxdradown_modelo}")
    print()


    # REGRESIÓN LOGÍSTICA (SE DEFINE Y ENTRENA UNA REGRESIÓN LOGÍSTICA, Y NOS QUEDAMOS CON EL OUTPUT Y LAS PREDICCIONES)
    modelo_regresion = regresion_logistica(datos_inputs, datos_output)
    y_hat_regresion = modelo_regresion.predict(datos_inputs_esc)

    # BACKTESTING REGRESION LOGISTICA
    serie_regresion, comision_regresion, diferencias_regresion, ratio_sharpe_reg, ratio_sortino_reg, drawdown, maxdradown_reg = backtesting(activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, y_hat_regresion, dias_reb, etiquetas_ic)

    # Calculamos los retornos acumulados y guardamos un DataFrame con todas las ordenes de compra
    df_retornos = serie_regresion.pct_change()
    serie_regresion = (1 + df_retornos).cumprod()
    df_diferencias = pd.concat(diferencias, axis=1).transpose()
    df_diferencias_regresion = pd.concat(diferencias_regresion, axis=1).transpose()

    # Se grafica la rentabilidad acumulada, las predicciones, el drawdown y los resultados
    plt.figure()
    plt.plot(serie_regresion, label='Backtesting Modelo "Regresion Logistica"')
    plt.title('Backtesting Regresion')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()

    plt.figure()
    plt.plot(y_hat_regresion)
    plt.title('Predicciones Regresion Logistica')
    plt.xlabel('epoch')
    plt.ylabel('predicciones')

    plt.figure()
    plt.plot(drawdown, label='Drawdown Modelo Regresion Logistica')
    plt.title('Drawdown Modelo Regresion Logistica')
    plt.xlabel('Fecha')
    plt.ylabel('Porcentaje')
    plt.legend()

    print(f"La rentabilidad acumulada del modelo Regresion Logistica es: {round(serie_regresion.iloc[-1] * 100, 2)}%")
    print(f"La comision total del modelo Regresion Logistica es: {comision_regresion}")
    print(f"El ratio de Sharpe del modelo Regresion Logistica es: {ratio_sharpe_reg}")
    print(f"El ratio de Sortino del modelo Regresion Logistica es: {ratio_sortino_reg}")
    print(f"El Maximo drawdown del modelo Regresion Logistica es: {maxdradown_reg}")
    print()


    # BENCHMARK SINTETICO - EW de las estrategias
    # Se crea un benchamk sintetico que hace todos los meses un Equal Wight de las estratgeias y el cash, se grafican los resultados
    serie_benchmark, comision_benchmark, diferencias, ratio_sharpe_bench, ratio_sortino_bench, drawdown, maxdradown_bench = benchmark(activos_close,activos_volumen,n_activos,capital_inicial,com, activos_open, dias_reb, etiquetas_ic)

    df_retornos = serie_benchmark.pct_change()
    serie_benchmark = (1 + df_retornos).cumprod()

    # Se grafica la rentabilidad acumulada, las predicciones, el dradown y los resultados
    plt.figure()
    plt.plot(serie_benchmark)
    plt.title('Backtesting Benchmark')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')

    plt.figure()
    plt.plot(drawdown, label = 'Drawdown Benchmark')
    plt.title('Drawdown Benchmark')
    plt.xlabel('Fecha')
    plt.ylabel('Porcentaje')
    plt.legend()

    print(f"La rentabilidad acumulada del benchmark es: {round(serie_benchmark.iloc[-1]*100,2)}%")
    print(f"La comision total del benchmark es: {comision_benchmark}")
    print(f"El ratio de Sharpe del benchmark es: {ratio_sharpe_bench}")
    print(f"El ratio de Sortino del benchmark es: {ratio_sortino_bench}")
    print(f"El Maximo drawdown del benchmark es: {maxdradown_bench}")

    # MONOS - PRUEBA DE ALEATORIEDAD   100 MONOS SE DEMORAN COMO 13 MINUTOS
    serie_monos = []
    comisiones_monos = []
    for i in range(10):
        serie_monos_int, comision_monos, diferencias = monos(
            activos_close, activos_volumen, n_activos, capital_inicial, com, activos_open, dias_reb, etiquetas_ic
        )
        serie_monos.append(serie_monos_int)
        comisiones_monos.append(comision_monos)
    serie_monos = pd.DataFrame(serie_monos).transpose()
    serie_monos.set_index(activos_close.index[54:], inplace=True)
    comisiones_monos = pd.DataFrame(comisiones_monos).transpose()
    plt.figure()
    plt.plot(serie_monos)
    plt.title('Backtesting Prueba Aleatoriedad')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')

    # Se calcula el mono 50 y el 100 para compararlos con los modelos
    ranking_monos = serie_monos.iloc[-1, :].sort_values(ascending=False)
    fila_central = (ranking_monos.shape[0] - 1) // 2
    mono_50 = ranking_monos[ranking_monos.index[fila_central]]
    comision_50 = comisiones_monos.iloc[:, ranking_monos.index[fila_central]]
    mono_100 = ranking_monos[ranking_monos.index[0]]
    comision_100 = comisiones_monos.iloc[:, ranking_monos.index[0]]

    mono_1 = ranking_monos[ranking_monos.index[-1]]
    comision_1 = comisiones_monos.iloc[:, ranking_monos.index[-1]]    

    # Se calculan los retonos de los monos seleccionados 
    # Mono 50%
    df_retornos = serie_monos.iloc[:, ranking_monos.index[fila_central]].pct_change()
    serie_monos.iloc[:, ranking_monos.index[fila_central]] = (1 + df_retornos).cumprod()
    # Mono 100%
    df_retornos = serie_monos.iloc[:, ranking_monos.index[0]].pct_change()
    serie_monos.iloc[:, ranking_monos.index[0]] = (1 + df_retornos).cumprod()

    valor = serie_backtesting.iloc[-1] * 44444  # Obtener el valor de la tercera columna en la primera fila
    percentil = percentileofscore(ranking_monos, valor)
    print(percentil)

    # Se grafican las series de precios de los 3 modelos, el benchmark y el mono del 50%
    plt.figure()
    plt.plot(serie_benchmark, label="benchmark sintetico")
    #plt.plot(serie_red, label="Red Neuronal")
    plt.plot(serie_regresion, label="Regresion logistica")
    plt.plot(serie_backtesting, label="Modelo")
    plt.plot(serie_monos.iloc[:, ranking_monos.index[fila_central]], label="Mono 50")
    plt.plot(serie_monos.iloc[:, ranking_monos.index[0]], label="Mono 100")
    plt.title('Backtesting')
    plt.xlabel('fecha')
    plt.ylabel('precio')
    plt.legend()

    # Se concatenan las comisiones de todo lo anterior y se grafican
    plt.figure()
    categorias = ['Modelo', 'Regresion', 'Benchmark', 'Mono 50%', 'Mono 100%']
    valores = [comision_backtesting, comision_regresion, comision_benchmark, comision_50.iloc[0], comision_100.iloc[0]]
    colores = plt.cm.Set1(np.linspace(0, 1, len(categorias)))
    fig, ax = plt.subplots()
    ax.bar(categorias, valores, color=colores, width=0.1)
    ax.tick_params(axis='x', labelsize=8)
    plt.title('Comisiones')
    plt.show()