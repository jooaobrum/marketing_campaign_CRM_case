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
class ModelClusterCfg:
    if not os.path.exists('../../../output'):
        # Create the folder if it doesn't exist
        os.makedirs('../../../output')

    model_cluster_infos = os.path.join("../../../models", "cluster_pipeline.json")
    model_cluster_filepath = os.path.join("../../../models", "cluster_pipeline.pkl")
    data_filepath = os.path.join("../../../artifacts", "processed_data.csv")
    output_inference = "../../../output/"

class ClusterModelInference:
    def __init__(self):
        self.model_loader_config = ModelClusterCfg()

    def initiate_cluster_inference(self):
        logging.info("Model Clustering Inference Started...")
        try:
            now = datetime.datetime.now()
            # Read the model to cluster data
            model = pd.read_pickle(self.model_loader_config.model_cluster_filepath)
            model_info = pd.read_json(self.model_loader_config.model_cluster_infos)
            logging.info("Loading completed...")

            

            # Read data to score
            df = pd.read_csv(self.model_loader_config.data_filepath)
            logging.info("Processed data read...")

            # Scoring 
            y_pred_prob = model["model"].predict_proba(df[model_info['features']])

            prob_clusters = []
            for customer in y_pred_prob:
                prob_clusters.append(create_order(customer, model["model"].classes_, k = 3))

            # Add ID
            for i in range((len(prob_clusters))):
                prob_clusters[i].insert(0, df['id'].values[i])
                prob_clusters[i].insert(0, now.strftime("%d/%m/%Y"))
            logging.info("Scoring done...")

            df_clusters = pd.DataFrame(prob_clusters, columns = ['ID', 'Date', 'Main Cluster', 'Alternative Cluster 1', 'Alternative Cluster 2'])
            df_clusters.to_csv(self.model_loader_config.output_inference + 'cluster_inference_' + now.strftime("%d%m%Y") + '.csv')
            logging.info("Saving output...")
        
        except Exception as e:
            raise CustomException(e, sys)