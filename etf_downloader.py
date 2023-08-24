import investpy
import yfinance as yf
import os
import pandas as pd


class ETFDataDownloader:
    def __init__(self, country='united states', period='8y'):
        self.country = country
        self.period = period

    def get_usa_etf_names(self):
        usa_etf_data = investpy.etfs.get_etfs(country=self.country)
        usa_etf_names = usa_etf_data['symbol'].tolist()
        return usa_etf_names

    def download_etf_data(self, etf_names):
        etf_symbols = ' '.join(etf_names)
        tit = yf.download(tickers=etf_symbols, period=self.period)
        threshold = 0.002  # Establecemos un threshold de un 2% para eliminar cualquier ETF que tenga muchos NA
        missing_values_percent = tit.Open.isna().mean()
        tit_close = tit.Close.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1)  # Nos quedamos solo con los ETF que su % de NA no supere el 2%
        tit_close = tit_close.dropna(axis='columns')
        tit_close.isna().sum().sum()
        #tit_close = tit.Close.dropna(axis='columns')
        return tit_close
    
    def download_etf_data_volumen(self, etf_names):
        etf_symbols = ' '.join(etf_names)
        tit = yf.download(tickers=etf_symbols, period=self.period)
        threshold = 0.002  # Establecemos un threshold de un 2% para eliminar cualquier ETF que tenga muchos NA
        missing_values_percent = tit.Open.isna().mean()
        tit_volume = tit.Volume.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1)  # Nos quedamos solo con los ETF que su % de NA no supere el 2%
        tit_volume = tit_volume.dropna(axis='columns')
        tit_volume.isna().sum().sum()
        #tit_volume = tit.Volume.dropna(axis='columns')
        return tit_volume
    
    def download_etf_data_open(self, etf_names):
        etf_symbols = ' '.join(etf_names)
        tit = yf.download(tickers=etf_symbols, period=self.period)
        threshold = 0.002  # Establecemos un threshold de un 2% para eliminar cualquier ETF que tenga muchos NA
        missing_values_percent = tit.Open.isna().mean()
        tit_open = tit.Open.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1)  # Nos quedamos solo con los ETF que su % de NA no supere el 2%
        tit_open = tit_open.dropna(axis='columns')
        tit_open.isna().sum().sum()
        #tit_open = tit.Open.dropna(axis='columns')
        return tit_open

    #def save_etf_data(self, data, output_dir):
    #    if not os.path.exists(output_dir):
    #        os.makedirs(output_dir)

    #    dataframes_columnas = {}
    #    for columna in data.columns:
    #        columna_df = data[[columna]].copy()
    #        columna_df.set_index(data.index, inplace=True)
    #        dataframes_columnas[columna] = columna_df

    #    for nombre_columna, columna_df in dataframes_columnas.items():
    #        nombre_archivo = os.path.join(output_dir, "{}.csv".format(nombre_columna))
    #        columna_df.to_csv(nombre_archivo)
    #        print("DataFrame de la columna '{}' guardado en '{}'".format(nombre_columna, nombre_archivo))


if __name__ == "__main__":
    downloader = ETFDataDownloader()

    usa_etf_names = downloader.get_usa_etf_names()
    print("Descargando datos de ETFs de Estados Unidos...")
    etf_data = downloader.download_etf_data_open(usa_etf_names)

    #directorio_datos_etfs = "datos_etfs"
    #downloader.save_etf_data(etf_data, directorio_datos_etfs)
    #print("Datos guardados en el directorio '{}'".format(directorio_datos_etfs))

