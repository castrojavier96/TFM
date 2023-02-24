import matplotlib.pyplot as plt
# funcion que, calcula rendimientos y hace el grafico de todas las estrategias
def graficar_series_precio(df1, df2, df3, df4):
    df_retornos = df1.pct_change()
    df1 = (1 + df_retornos).cumprod()

    df_retornos = df2.pct_change()
    df2 = (1 + df_retornos).cumprod()

    df_retornos = df3.pct_change()
    df3 = (1 + df_retornos).cumprod()

    df_retornos = df4.pct_change()
    df4 = (1 + df_retornos).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df1, label='Serie Momentum')
    ax.plot(df2, label='Serie EW')
    ax.plot(df3, label='Serie Volmax')
    ax.plot(df4, label='Serie VolMin')
    
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title('Rentabilidad Acumulada')
    
    ax.legend()
    
    return plt.show()