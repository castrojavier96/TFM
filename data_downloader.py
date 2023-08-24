#pip install python-dotenv fredapi


import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

class DataDownloader:
    def __init__(self):
        load_dotenv()  # Cargar variables de entorno desde .env
        self.api_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=self.api_key)

    def download_series(self, series_ids, start_date, end_date):
        try:
            data = {}
            for series_id in series_ids:
                series_data = self.fred.get_series(series_id, start_date, end_date)
                data[series_id] = series_data
            return data
        except Exception as e:
            return f"Error downloading series: {str(e)}"

if __name__ == "__main__":
    downloader = DataDownloader()
    
    # Ejemplo de descarga de datos
    #series_ids = ["GDPC1", "CPILFESL", "PCEPI", "PPIFID", "DFEDTARU", "CSUSHPISA", "UNRATE"]
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
    for series_id, data in downloaded_data.items():
        print(f"Series ID: {series_id}")
        print(data)
        print("=" * 30)

    # Crear un DataFrame con los datos descargados
    df = pd.DataFrame(downloaded_data, index=pd.date_range(start_date, end_date))
    df = df.fillna(method='ffill')
    print(df)