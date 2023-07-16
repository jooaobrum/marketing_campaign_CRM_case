import os
import sys

from utils import CustomException
from logger import logging
import pandas as pd

from dataclasses import dataclass


@dataclass
class DataIngestionCfg():
    raw_data_path: str = os.path.join('../../artifacts', 'data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_cfg = DataIngestionCfg()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started...")
        try:
            df = pd.read_csv('../../data/ml_project1_data.csv')
            logging.info("Dataset read...")

            # Saving to the raw path
            df.to_csv(self.ingestion_cfg.raw_data_path, header = True, index = False)
            logging.info("Raw ingested...")
        
            logging.info("Ingestion completed...")

            return self.ingestion_cfg.raw_data_path
                
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()

    obj.initiate_data_ingestion()

   