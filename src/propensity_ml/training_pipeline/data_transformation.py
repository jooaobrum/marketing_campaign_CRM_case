import os
import sys

from utils import CustomException
from logger import logging
from etl import transform
import pandas as pd

from dataclasses import dataclass




@dataclass
class DataTransformationCfg():
    raw_data_path: str = os.path.join('../../../artifacts/propensity', 'data.csv')
    processed_data_path: str = os.path.join('../../../artifacts/propensity', 'processed_data.csv')

class DataTransformation():
    def __init__(self):
        self.transform_cfg = DataTransformationCfg()


    def initiate_data_transformation(self):
            logging.info("Data Transformation Started...")
            try:
                df = pd.read_csv(self.transform_cfg.raw_data_path)
                logging.info("Dataset from artifacts read...")

                # Rename Columns
                df = transform.rename_columns(df)
                logging.info("Columns renamed...")
                
                # Cast Columns
                df = transform.cast_columns(df)
                logging.info("Columns casted...")
                
                # Create columns
                df = transform.feature_engineering(df)
                logging.info("Features created...")


                # Filter columns
                df = transform.filter_columns(df)
                logging.info("Features filtered...")

                # Saving the processed dataset in the artifacts
                df.to_csv(self.transform_cfg.processed_data_path)

                return self.transform_cfg.processed_data_path
            except Exception as e:
                raise CustomException(e, sys)

