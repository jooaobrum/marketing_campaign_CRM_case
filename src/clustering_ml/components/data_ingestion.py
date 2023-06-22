import os
import sys
#sys.path.append("G:\\Mon Drive\\Projetos Pessoais\\Projetos de Portfolio\\CRM Ifood - nao concluido\\ifood_marketing_campaign")
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionCfg():
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_cfg = DataIngestionCfg()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started...")
        try:
            df = pd.read_csv('data\ml_project1_data.csv')
            logging.info("Dataset read...")

            os.makedirs(os.path.dirname(self.ingestion_cfg.train_data_path), exist_ok = True)
            
            # Saving to the raw path
            df.to_csv(self.ingestion_cfg.raw_data_path, header = True, index = False)
            logging.info("Raw ingested...")


            
            # Split data
            logging.info("Split Started...")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            

            # Saving to the  training path
            train_set.to_csv(self.ingestion_cfg.train_data_path, header = True, index = False)
            logging.info("Train ingested")

            # Saving to the  testing path
            test_set.to_csv(self.ingestion_cfg.test_data_path, header = True, index = False)
            logging.info("Test ingested")
        
            logging.info("Ingestion completed...")

            return (
                self.ingestion_cfg.train_data_path,
                self.ingestion_cfg.test_data_path

            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()

    obj.initiate_data_ingestion()

   