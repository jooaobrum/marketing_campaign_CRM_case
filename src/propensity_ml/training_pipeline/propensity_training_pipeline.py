import data_ingestion
import data_transformation
import model_trainer


if __name__ == '__main__':
    ingestor = data_ingestion.DataIngestion()
    transformer = data_transformation.DataTransformation()
    trainer = model_trainer.ModelTrainer()
    
    
    # Extract
    ingestor.initiate_data_ingestion()

    # Transform
    transformer.initiate_data_transformation()

    # Training
    trainer.initiate_model_training()