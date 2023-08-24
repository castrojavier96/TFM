from data_downloader import DataDownloader
from etf_downloader import ETFDataDownloader
from s3_uploader import S3Uploader
import pandas as pd
import numpy as np
import os



if __name__ == "__main__":
    downloader = DataDownloader()
    uploader = S3Uploader()
    downloader_etf = ETFDataDownloader()
    
    # Ejemplo de descarga de datos
    series_ids = [
        "GDPC1", 
        "CPILFESL", 
        "PCEPI", 
        "PPIFID", 
        "DFEDTARU", 
        "CSUSHPISA", 
        "UNRATE",
        'SP500', # S&P 500
        'DGS10', # 10-Year Treasury Constant Maturity Rate
        'T10Y2Y', # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        'T10Y3M', # 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
        'T10YIE',
        'FEDFUNDS', # Effective Federal Funds Rate
        'UNRATE', # Unemployment Rate
        'CPIAUCSL', # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
        'CPILFESL', # Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average
        'PCEPI', # Personal Consumption Expenditures: Chain-type Price Index
        'M2', # M2 Money Stock
        'M2V', # Velocity of M2 Money Stock
        'BASE', # St. Louis Adjusted Monetary Base
        'USSLIND', # Leading Index for the United States
        'UMCSENT', # University of Michigan: Consumer Sentiment
        'IPMAN', # Industrial Production: Manufacturing (NAICS)
        'HOUST', # Housing Starts: Total: New Privately Owned Housing Units Started
        'GS10', # 10-Year Treasury Constant Maturity Rate
        'BAA10Y', # Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity
        'TEDRATE', # TED Spread
        'WALCL', # Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level
        'GDP',
        'GDPC1',
        'CPILFESL',
        'DTWEXB',
        'BAMLH0A0HYM2',
        'DGS2',
        'A191RP1Q027SBEA',
        'VIXCLS'
        ]
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    downloaded_data = downloader.download_series(series_ids, start_date, end_date)
    
    # Crear un DataFrame con los datos descargados
    df = pd.DataFrame(downloaded_data, index=pd.date_range(start_date, end_date))
    df = df.fillna(method='ffill')
    df = df.iloc[1326:]

    # Guardar DataFrame en un archivo CSV
    df.to_csv('macro_data.csv', index=True)

    
    # Se descargan los datos de los ETFs
    usa_etf_names = downloader_etf.get_usa_etf_names()
    print("Descargando datos de ETFs de Estados Unidos...")
    etf_data = downloader_etf.download_etf_data(usa_etf_names)
    etf_data_volumen = downloader_etf.download_etf_data_volumen(usa_etf_names)
    etf_data_open = downloader_etf.download_etf_data_open(usa_etf_names)

    # Guardar DataFrame en un archivo CSV
    etf_data.to_csv('etf_data.csv', index=True)
    etf_data_volumen.to_csv('etf_data_volumen.csv', index=True)
    etf_data_open.to_csv('etf_data_open.csv', index=True)

    # Se guardan los datos en archivos CSV
    #directorio_datos_etfs = "datos_etfs"
    #downloader_etf.save_etf_data(etf_data, directorio_datos_etfs)
    #print("Datos guardados en el directorio '{}'".format(directorio_datos_etfs))
    

    # Cargar archivo CSV a S3 en la carpeta 'macro'
    uploader.upload_file('macro_data.csv', 'macro/macro_data.csv')

    # Cargar archivo de ETFs en CSV a S3 en la carpeta 'ETF'
    uploader.upload_file('etf_data.csv', 'ETF/etf_data.csv')

    # Cargar archivo de ETFs en CSV a S3 en la carpeta 'ETF'
    uploader.upload_file('etf_data_volumen.csv', 'ETF/etf_data_volumen.csv')

    # Cargar archivo de ETFs en CSV a S3 en la carpeta 'ETF'
    uploader.upload_file('etf_data_open.csv', 'ETF/etf_data_open.csv')