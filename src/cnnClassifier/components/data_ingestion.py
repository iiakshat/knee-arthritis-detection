import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            url = self.config.source_URL
            dwn_dir = self.config.local_data_file
            os.makedirs('artifacts/data_ingestion', exist_ok=True)
            logger.info('Downloading file from %s to %s', url, dwn_dir)

            file_id = url.split('/')[-2]
            download_url = f"https://drive.google.com/uc?/export=download&id={file_id}"

            gdown.download(download_url, dwn_dir )

            logger.info('Download completed')
        
        except Exception as e:
            raise e
    
    def extractor(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as f:
            f.extractall(unzip_path)