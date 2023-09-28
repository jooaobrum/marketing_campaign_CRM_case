import os
import sys

from utils import CustomException
from logger import logging
import pandas as pd
import datetime 

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

class PropensityModelInference:
    def __init__(self):
        self.model_loader_config = ModelPropensityCfg()

    def initiate_propensity_inference(self):
        logging.info("Model Propensity Inference Started...")
        try:
            now = datetime.datetime.now()
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
            logging.info("Saving output...")
        
        except Exception as e:
            raise CustomException(e, sys)