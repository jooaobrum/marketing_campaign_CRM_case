import os
import sys

from utils import CustomException
import utils
from logger import logging
import pandas as pd
import boto3

from dataclasses import dataclass


@dataclass
class DataIngestionCfg():
    if not os.path.exists('artifacts/propensity'):
        # Create the folder if it doesn't exist
        os.makedirs('artifacts/propensity')
    raw_data_path: str = os.path.join('artifacts/propensity', 'data.csv')
    
    bucket_name: str = 'jooaobrum-projects'
    s3_folder_path: str = 'crm-project/data'
    s3_client = boto3.client('s3')

class DataIngestion():
    def __init__(self):
        self.ingestion_cfg = DataIngestionCfg()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started...")
        try:
            file_name = 'ml_project1_data.csv'
            
            df = utils.read_file_from_s3(self.ingestion_cfg.s3_client, self.ingestion_cfg.bucket_name, os.path.join(self.ingestion_cfg.s3_folder_path,file_name))
            #df = pd.read_csv('data/ml_project1_data.csv')
            logging.info("Dataset read...")

            # Saving to the raw path
            df.to_csv(self.ingestion_cfg.raw_data_path, header = True, index = False)
            logging.info("Raw ingested...")
        
            logging.info("Ingestion completed...")

            return self.ingestion_cfg.raw_data_path
                
        
        except Exception as e:
            raise CustomException(e, sys)

