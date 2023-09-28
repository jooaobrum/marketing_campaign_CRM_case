
# Standard Libraries
import os
import sys
import datetime
import json
from dataclasses import dataclass
from warnings import filterwarnings

# External Libraries
import pandas as pd
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
import scikitplot as skplt
from xgboost import XGBClassifier


# Custom Libraries
from utils import CustomException
from logger import logging
from etl import transform

# Scikit-Learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeClassifier







@dataclass
class ModelTrainingCfg:

    if not os.path.exists('../../../models'):
        # Create the folder if it doesn't exist
        os.makedirs('../../../models')

    propensity_infos_filepath = "../../../artifacts/propensity"
    propensity_pipeline_filepath = os.path.join("../../../models", "propensity_pipeline.pkl")
    processed_data_filepath = os.path.join("../../../artifacts/propensity", "processed_data.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingCfg()

    def initiate_model_training(self):
        logging.info("Model Propensity/Training Started...")
        try:

            # %%
            df = pd.read_csv(self.model_trainer_config.processed_data_filepath)
            logging.info("Processed data read...")

            features = ['income','recency', 'mntwines', 'mntfruits', 'mntmeatproducts', 
                        'mntsweetproducts', 'mntgoldprods', 'numdealspurchases',
                        'numwebpurchases', 'numcatalogpurchases', 'numstorepurchases',
                        'numwebvisitsmonth', 'loyalty_days', 'loyalty_months',
                        'amount_spent', 'amount_spent_month', 'total_purchases',
                        'age', 'purchases_month','mntfishproducts',
                        'percentage_spent_wines', 'percentage_spent_fruits',
                        'percentage_spent_meat', 'percentage_spent_fish',
                        'percentage_spent_sweet', 'percentage_spent_gold',
                        'percentage_type_deals', 'percentage_type_web',
                        'percentage_type_catalog', 'percentage_type_store', 'complain',
                        'kidhome', 'teenhome', 'education', 'marital_status']

            target = 'response'

            # Split in train and test
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.15)
            logging.info("Split in train and test...")

            
            # Features to use imputer and scaler
            numeric_features = ['income','recency', 'mntwines', 'mntfruits', 'mntmeatproducts', 
                                'mntsweetproducts', 'mntgoldprods', 'numdealspurchases',
                                'numwebpurchases', 'numcatalogpurchases', 'numstorepurchases',
                                'numwebvisitsmonth', 'loyalty_days', 'loyalty_months',
                                'amount_spent', 'amount_spent_month', 'total_purchases',
                                'age', 'purchases_month','mntfishproducts',
                                'percentage_spent_wines', 'percentage_spent_fruits',
                                'percentage_spent_meat', 'percentage_spent_fish',
                                'percentage_spent_sweet', 'percentage_spent_gold',
                                'percentage_type_deals', 'percentage_type_web',
                                'percentage_type_catalog', 'percentage_type_store', 'complain']

            # Features to conver to ordinal encoding
            categorical_features = ['kidhome', 'teenhome', 'education', 'marital_status']

            # Numerical transformations
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median"))]
            )

            # Categorical transformations
            categorical_transformer = Pipeline(
                steps=[
                    ("encoder", TargetEncoder())
            ]
            )

            # Preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features, ),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder='passthrough'
                
            )

            
            feat_selector_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', SequentialFeatureSelector(DecisionTreeClassifier()))
            ])

            feat_selector_pipe.fit(X_train, y_train)

            selected_features = feat_selector_pipe.get_feature_names_out()
            selected_features


            feat_selector_pipe.fit(X_train, y_train)
            logging.info("Selecting best features...")
            

            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))


            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            param_grid = {
                'learning_rate': [0.01],
                'max_depth': [3],
                'n_estimators': [500],
                'objective' : ['binary:logistic'],
                'scale_pos_weight' : [class_weight_dict[1]]
            }

            model = XGBClassifier()

            # Grid Search for hyperparameters tuning
            grid_search = GridSearchCV( model, 
                                        param_grid,
                                        n_jobs = -1,
                                        cv = skf,
                                        scoring = 'recall',
                                        verbose = 3,
                                        refit = True)

            # Pipeline de dados
            model_pipe = Pipeline( steps = [('preprocessor', preprocessor),
                                        ('selector', SequentialFeatureSelector(DecisionTreeClassifier())),
                                        ('gridsearch', grid_search)])

            # Fit treino
            model_pipe.fit(X_train, y_train)
            logging.info("Fitting best model with grid serch and xgboost...")

            # Best threshold according to the experiments (from notebook)
            best_th = 0.4
            
            # Predictions
            y_prob_train = model_pipe.predict_proba(X_train)
            y_prob_test = model_pipe.predict_proba(X_test)

            y_pred_train = (y_prob_train[:, 1] > best_th).astype(int)
            y_pred_test = (y_prob_test[:, 1] > best_th).astype(int)


            train_acc_score = accuracy_score(y_train, y_pred_train)
            test_acc_score = accuracy_score(y_test, y_pred_test)

            train_precision = precision_score(y_train, y_pred_train)
            test_precision = precision_score(y_test, y_pred_test)

            train_recall = recall_score(y_train, y_pred_train)
            test_recall = recall_score(y_test, y_pred_test)

            train_f1_score = f1_score(y_train, y_pred_train)
            test_f1_score = f1_score(y_test, y_pred_test)
            logging.info("Scoring train and test...")

            logging.info(f"Accuracy for the training: {train_acc_score:.2f}")
            logging.info(f"Accuracy for the test: {test_acc_score:.2f}")

            logging.info(f"Precision for the training: {train_precision:.2f}")
            logging.info(f"Precision for the test: {test_precision:.2f}")

            logging.info(f"Recall for the training: {train_recall:.2f}")
            logging.info(f"Recall for the test: {test_recall:.2f}")

            logging.info(f"F1 Score for the training: {train_f1_score:.2f}")
            logging.info(f"F1 Score for the test: {test_f1_score:.2f}")


            model_info = pd.Series({
                "goal": "propensity_to_accept_cmp",
                "model":model_pipe,
                "features":numeric_features + categorical_features,
                "train_recall_score":train_recall,
                "test_recall_score":test_recall,
                "path": self.model_trainer_config.propensity_pipeline_filepath,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %Hh-%Mm-%Ss")

            })

            model_info.to_pickle(self.model_trainer_config.propensity_pipeline_filepath)
            
            model_info_json = model_info.drop('model')
            model_info_json.to_json(os.path.splitext(self.model_trainer_config.propensity_pipeline_filepath)[0] + ".json", indent = 4)
            logging.info("Model saved...")

            return model_info
        except Exception as e:
            raise CustomException(e, sys)
        

