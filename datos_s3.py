# Script que lee los datos desde un bucket de S3 y obtiene dataframes de Open, Close y Volumen

#Importamos librerias
import boto3
import pandas as pd
from io import StringIO

# funcion para obtener los Datos en 3 Dataframe de opne, close y volumen
def obtener_datos():

    # Se dan las claves de acceso a aws
    access_key_id = 'AKIAWMUUWTYFIT3U6WMF'
    secret_access_key = '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'

    # Inicia la sesión de boto3
    s3 = boto3.client('s3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key)

    # Nombre del bucket y prefijo para los archivos CSV
    bucket_name = 'tfmrebalanceador'

    # Obtiene la lista de los objetos del bucket
    response = s3.list_objects_v2(Bucket=bucket_name)

    # Lista para almacenar todos los DataFrames

    li = []
    li_volumen = []
    li_open = []
    new_li_names = []
    li_names = []
    # Iterar sobre cada objeto en el bucket y leer los archivos CSV
    for obj in response['Contents']:
        if obj['Key'].endswith('.csv') and obj['Key']!='inflacion.csv' and obj['Key']!='PIB_EU.csv':
            # Lee el archivo CSV desde S3
            file_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            file_content = file_obj['Body'].read().decode('utf-8')
            # Lee el DataFrame a partir del archivo CSV
            df = pd.read_csv(StringIO(file_content), index_col='Date', header=0) # leemos los archivos y seteamos el Date como indice
            df_close = df.loc['2013-01-01':'2022-11-01',['Close']]# nos quedamos con los datos desde el 2013 a Noviembre 2022
            df_volumen = df.loc['2013-01-01':'2022-11-01',['Volume']]
            df_open = df.loc['2013-01-01':'2022-11-01',['Open']]
            li.append(df_close)
            li_volumen.append(df_volumen)
            li_open.append(df_open)
            new_li_names.append(pd.read_csv(StringIO(file_content),sep="|"))
            li_names.append(obj['Key'].split(".")[0][-4:])

    frame = pd.concat(li, axis=1, ignore_index=True) # concatenamos en un dataframe 
    frame.sort_values(by=['Date'],inplace=True)# ordenamos los datos para que queden por fechas
    frame.columns = li_names # le ponemos los nombre de los ETF a cada columna

    threshold = 0.02# establezemos un treshold de un 2% para eliminar cualquier ETF que tenga muchos NA
    missing_values_percent = frame.isna().mean()
    frame = frame.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1) # nos quedamos solo con los ETF que su % de NA no supere el 2%
    print(frame.isna().sum().sum())
    frame.fillna(method='ffill', limit=2,inplace=True)# rellenamos un maximo de 2 NA hacia delante con el metodo de forward fill
    print(frame.isna().sum().sum())

    datos = frame.dropna(axis=1)# El resto de ETF que tienen un NA los eliminamos
    datos.isna().sum().sum()
    datos.index = pd.to_datetime(datos.index) # pasamos el indice a un formato datetime

    # Este mismo proceso lo hacemos para los datos de volumen 
    frame_volumen = pd.concat(li, axis=1, ignore_index=True)
    frame_volumen.sort_values(by=['Date'],inplace=True)
    frame_volumen.columns = li_names

    missing_values_percent = frame_volumen.isna().mean()
    frame_volumen = frame_volumen.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1)

    frame_volumen.fillna(method='ffill', limit=2, inplace=True)

    datos_volumen = frame_volumen.dropna(axis=1)
    datos_volumen.isna().sum().sum()
    datos_volumen.index = pd.to_datetime(datos_volumen.index)

    # Este mismo proceso lo hacemos para los datos de Open
    frame_open = pd.concat(li, axis=1, ignore_index=True)
    frame_open.sort_values(by=['Date'],inplace=True)
    frame_open.columns = li_names

    missing_values_percent = frame_open.isna().mean()
    frame_open = frame_open.drop(missing_values_percent[missing_values_percent > threshold].index, axis=1)

    frame_open.fillna(method='ffill', limit=2, inplace=True)

    datos_open = frame_open.dropna(axis=1)
    datos_open.isna().sum().sum()
    datos_open.index = pd.to_datetime(datos_open.index)

    return datos, datos_volumen, datos_open # Se puede ver que los 3 Dataframe terminan con 56 activos y 2524 dias


def obtener_macros():
    # Se dan las claves de acceso a aws
    access_key_id = 'AKIAWMUUWTYFIT3U6WMF'
    secret_access_key = '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'

    # Inicia la sesión de boto3
    s3 = boto3.client('s3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key)

    # Nombre del bucket y prefijo para los archivos CSV
    bucket_name = 'tfmrebalanceador'

    # Obtiene la lista de los objetos del bucket
    response = s3.list_objects_v2(Bucket=bucket_name)


    # Iterar sobre cada objeto en el bucket y leer los archivos CSV
    for obj in response['Contents']:
        if obj['Key']=='inflacion.csv':
            # Lee el archivo CSV desde S3
            file_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            file_content = file_obj['Body'].read().decode('utf-8')
            # Lee el DataFrame a partir del archivo CSV
            inflacion = pd.read_csv(StringIO(file_content), index_col='Fecha', header=0) # leemos los archivos y seteamos el Date como indice
    
    inflacion.index = pd.to_datetime(inflacion.index)        
    inflacion.sort_values(by=['Fecha'],inplace=True)
    inflacion = inflacion.iloc[:,[0,3]]

    for obj in response['Contents']:
        if obj['Key']=='PIB_EU.csv':
            # Lee el archivo CSV desde S3
            file_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            file_content = file_obj['Body'].read().decode('utf-8')
            # Lee el DataFrame a partir del archivo CSV
            pib = pd.read_csv(StringIO(file_content), header=0) # leemos los archivos y seteamos el Date como indice
    
    return inflacion, pib