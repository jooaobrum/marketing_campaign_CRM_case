import os
import sys

from utils import CustomException
import utils
from logger import logging
import pandas as pd
import datetime 
import boto3

from dataclasses import dataclass

# Organise the classes by the highest probability
def create_order(prob, classes, k = 5):
    list_probs = list(zip(prob, classes))
    
    list_probs.sort(key = lambda x: x[0], reverse = True)

    return [element[1] for element in list_probs[:k]]

@dataclass
class ModelPropensityCfg:
    if not os.path.exists('output'):
        # Create the folder if it doesn't exist
        os.makedirs('output')

    model_propensity_infos = os.path.join("models", "propensity_pipeline.json")
    model_propensity_filepath = os.path.join("models", "propensity_pipeline.pkl")
    data_filepath = os.path.join("artifacts/propensity", "processed_data.csv")
    output_inference = "output/"

    #AWS Bucket
    bucket_name = 'jooaobrum-projects'
    s3_file_path = 'crm-project/models'
    s3_client = boto3.client('s3')

class PropensityModelInference:
    def __init__(self):
        self.model_loader_config = ModelPropensityCfg()

    def initiate_propensity_inference(self):
        logging.info("Model Propensity Inference Started...")
        try:
            now = datetime.datetime.now()

            # Download model from S3 bucket
            utils.download_file_from_s3(self.model_loader_config.s3_client, 
                                        self.model_loader_config.bucket_name, 
                                        os.path.join(self.model_loader_config.s3_file_path, self.model_loader_config.model_propensity_filepath.split('/')[-1]),
                                        self.model_loader_config.model_propensity_filepath.split('/')[0])
            
            # Download model info from S3
            utils.download_file_from_s3(self.model_loader_config.s3_client, 
                                        self.model_loader_config.bucket_name, 
                                        os.path.join(self.model_loader_config.s3_file_path, self.model_loader_config.model_propensity_filepath.split('/')[-1].split('.')[0] + '.json'),
                                        self.model_loader_config.model_propensity_filepath.split('/')[0])
            logging.info("Downloading model from cloud...")


            # Read the model to score data
            model = pd.read_pickle(self.model_loader_config.model_propensity_filepath)
            model_info = pd.read_json(self.model_loader_config.model_propensity_infos)
            logging.info("Loading completed...")

            

            # Read data to score
            df = pd.read_csv(self.model_loader_config.data_filepath)
            logging.info("Processed data read...")

            # Scoring 
            y_pred_prob = model["model"].predict_proba(df[model_info['features']])

            # Add ID
            df_propensity = pd.DataFrame()
            df_propensity['ID'] = df['id'].values.tolist()
            df_propensity['Date'] = now.strftime("%d%/%m/%Y")
            df_propensity['Prob'] = y_pred_prob[:, 1]
            logging.info("Scoring done...")

            df_propensity = df_propensity.sort_values(["Date", "Prob"], ascending= [True, False]).reset_index(drop = True)
            df_propensity.to_csv(self.model_loader_config.output_inference + 'propensity_inference_' + now.strftime("%d%m%Y%Hh%Mm%Ss") + '.csv')
            logging.info("Saving output locally...")

            utils.upload_file_to_s3(self.model_loader_config.s3_client,
                                    self.model_loader_config.bucket_name,
                                    self.model_loader_config.output_inference + 'propensity_inference_' + now.strftime("%d%m%Y%Hh%Mm%Ss") + '.csv',
                                    os.path.join(self.model_loader_config.s3_file_path.split('/')[0], self.model_loader_config.output_inference + 'propensity_inference_' + now.strftime("%d%m%Y%Hh%Mm%Ss") + '.csv'))
            
            logging.info("Saving output on cloud...")

            os.remove(self.model_loader_config.output_inference + 'propensity_inference_' + now.strftime("%d%m%Y%Hh%Mm%Ss") + '.csv')
            logging.info("Removing output locally...")

            os.remove(self.model_loader_config.model_propensity_filepath)
            os.remove(self.model_loader_config.model_propensity_infos)
            logging.info("Removing model locally...")
        
        except Exception as e:
            raise CustomException(e, sys)