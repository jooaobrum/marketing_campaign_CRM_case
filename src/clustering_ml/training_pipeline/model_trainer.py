
import os
import sys
import datetime
from utils import CustomException
from logger import logging
from etl import transform
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from feature_engine import imputation
from sklearn.impute import SimpleImputer
from feature_engine import encoding
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt
from dataclasses import dataclass


def save_clustering_info(df, categories, cluster_id, image_label):
    pca_num_components = 2
    reduced_data = PCA(n_components=pca_num_components).fit_transform(df[categories].dropna())
    results = pd.DataFrame(reduced_data,columns=['PCA1','PCA2'])
    sns.heatmap(df[categories + [cluster_id]].groupby(cluster_id).mean(), cmap = 'YlGnBu', annot = True)
    plt.title('K-means Clustering with 2 dimensions')
    plt.savefig(image_label + ".png")
    plt.close()



@dataclass
class ModelTrainingCfg:

    if not os.path.exists('models'):
        # Create the folder if it doesn't exist
        os.makedirs('models')

    cluster_infos_filepath = "artifacts/clustering"
    cluster_pipeline_filepath = os.path.join("models", "cluster_pipeline.pkl")
    processed_data_filepath = os.path.join("artifacts/clustering", "processed_data.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingCfg()

    def initiate_model_training(self):
        logging.info("Model Clustering/Training Started...")
        try:

            # %%
            df = pd.read_csv(self.model_trainer_config.processed_data_filepath)
            logging.info("Processed data read...")


            # Scaling for clustering
            mms = MinMaxScaler()

            df_num = df.select_dtypes(include = ['float64', 'int64'])
            df_scaled = pd.DataFrame(mms.fit_transform(df_num), columns = df_num.columns.tolist())
            logging.info("Processed data scaled...")

            # Clustering products 
            categories_prod = ['percentage_spent_wines', 'percentage_spent_fruits', 'percentage_spent_meat', 'percentage_spent_fish', 'percentage_spent_sweet','percentage_spent_gold']
            kmeans_prod = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(df_scaled[categories_prod])
            cluster_id_prod = kmeans_prod.labels_

            df['cluster_id_prod'] = cluster_id_prod
            logging.info("Cluster for products created...")

            # Save heatmap of clustering
            save_clustering_info(df, categories_prod, "cluster_id_prod", self.model_trainer_config.cluster_infos_filepath + "/cluster_id_prod")

            # Clustering Channels 
            categories_channel = ['percentage_type_deals', 'percentage_type_web', 'percentage_type_catalog', 'percentage_type_store']
            kmeans_channel = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(df_scaled[categories_channel])
            cluster_id_channel = kmeans_channel.labels_

            df['cluster_id_channel'] = cluster_id_channel
            logging.info("Cluster for channels created...")

            # Save heatmap of clustering
            save_clustering_info(df, categories_channel, "cluster_id_channel", self.model_trainer_config.cluster_infos_filepath + "/cluster_id_channel")



            # Clustetering RFM
            categories_rfm = ['recency', 'total_purchases', 'amount_spent']
            kmeans_rfm = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(df_scaled[categories_rfm])

            cluster_id_rfm = kmeans_rfm.labels_

            df['cluster_id_rfm'] = cluster_id_rfm
            df_scaled['cluster_id_rfm'] = cluster_id_rfm
            save_clustering_info(df_scaled, categories_rfm, "cluster_id_rfm", self.model_trainer_config.cluster_infos_filepath + "/cluster_id_rfm")

            logging.info("Cluster for RFM created...")

            # Merge Clustering
            df['cluster_id_all'] = df['cluster_id_prod'].astype(str) + '_' + df['cluster_id_channel'].astype(str) + '_' + df['cluster_id_rfm'].astype(str)
            logging.info("All Clusters merged...")

            # Training a random forest to learn the clusters
            
            # Define the features of the model
            features = ['education', 'marital_status', 'income', 'kidhome',
                        'teenhome', 'recency', 'percentage_spent_wines', 'percentage_spent_fruits',
                        'percentage_spent_meat', 'percentage_spent_fish', 'percentage_spent_sweet',
                        'percentage_spent_gold', 'percentage_type_deals', 'percentage_type_web',
                        'percentage_type_catalog', 'percentage_type_store', 'numwebvisitsmonth',
                        'complain', 'loyalty_days', 'amount_spent', 'amount_spent_month',
                        'total_purchases', 'accepted_campaigns', 'age', 'purchases_month']

            # Define the target
            target = ['cluster_id_all']

            # Split train test
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.3, random_state = 42)
            logging.info("Train test split done...")

            # One Hot Encoding
            ohe_features = ['education', 'marital_status']
            onehot = encoding.OneHotEncoder(drop_last=True, variables = ohe_features)

            # Imputation of Income
            imput_features = 'income'
            imput_income = imputation.MeanMedianImputer(variables = imput_features)


            # Fit the Model
            rf = RandomForestClassifier(n_estimators = 100, class_weight='balanced_subsample')

            # Data Pipeline
            model_pipe = Pipeline( steps = [('Mean Imputation', imput_income),
                                            ('OHE', onehot),
                                            ('RF Model', rf)])

            
            # Training
            model_pipe.fit(X_train, y_train.values.ravel())
            logging.info("Random Forest fitted...")

            # Predictions
            y_pred_train = model_pipe.predict(X_train)
            y_pred_test = model_pipe.predict(X_test)

            train_acc_score = accuracy_score(y_train, y_pred_train)
            test_acc_score = accuracy_score(y_test, y_pred_test)
            logging.info(f"Accuracy for the training: {train_acc_score:.2f} ", )
            logging.info(f"Accuracy for the test: {test_acc_score:.2f}",)

            model_info = pd.Series({
                "goal": "clustering",
                "model":model_pipe,
                "features":features,
                "train_acc_score":train_acc_score,
                "test_acc_score":test_acc_score,
                "path": self.model_trainer_config.cluster_pipeline_filepath,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %Hh-%Mm-%Ss")

            })

            model_info.to_pickle(self.model_trainer_config.cluster_pipeline_filepath)
            
            model_info_json = model_info.drop('model')
            model_info_json.to_json(os.path.splitext(self.model_trainer_config.cluster_pipeline_filepath)[0] + ".json", indent = 4)
            logging.info("Model saved...")

            return model_info
        except Exception as e:
            raise CustomException(e, sys)
        

