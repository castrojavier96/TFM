import os
import boto3
from dotenv import load_dotenv
import pandas as pd
from io import StringIO

class S3Uploader:
    def __init__(self):
        load_dotenv()  # Cargar variables de entorno desde .env
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY")
        self.aws_secret_key = os.getenv("AWS_SECRET_KEY")
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

    def upload_file(self, local_file_path, remote_file_path):
        self.s3_client.upload_file(local_file_path, self.s3_bucket, remote_file_path)

    def download_file(self, folder_path):
        dataframes = []
        objects = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=folder_path)
        li = []
        li_volumen = []
        li_open = []
        new_li_names = []
        li_names = []
        # Iterar sobre cada objeto en el bucket y leer los archivos CSV
        for obj in objects['Contents']:
            if obj['Key'].endswith('.csv'):
                # Lee el archivo CSV desde S3
                file_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=obj['Key'])
                file_content = file_obj['Body'].read().decode('utf-8')
                # Lee el DataFrame a partir del archivo CSV
                df = pd.read_csv(StringIO(file_content), index_col='Date', header=0) # leemos los archivos y seteamos el Date como indice
                li.append(df)
                new_li_names.append(pd.read_csv(StringIO(file_content), sep="|"))
                li_names.append(obj['Key'].split(".")[0][-4:])
        return li
    
    def download_file_macro(self, folder_path):
        dataframes = []
        objects = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=folder_path)
        li = []
        li_volumen = []
        li_open = []
        new_li_names = []
        li_names = []
        # Iterar sobre cada objeto en el bucket y leer los archivos CSV
        for obj in objects['Contents']:
            if obj['Key'].endswith('.csv'):
                # Lee el archivo CSV desde S3
                file_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=obj['Key'])
                file_content = file_obj['Body'].read().decode('utf-8')
                # Lee el DataFrame a partir del archivo CSV
                df = pd.read_csv(StringIO(file_content), header=0) # leemos los archivos y seteamos el Date como indice
                li.append(df)
                new_li_names.append(pd.read_csv(StringIO(file_content), sep="|"))
                li_names.append(obj['Key'].split(".")[0][-4:])
        return li

if __name__ == "__main__":
    uploader = S3Uploader()
    # Ejemplo de carga de archivo a S3
    uploader.upload_file('archivo.csv', 'carpeta_remota/archivo.csv')
