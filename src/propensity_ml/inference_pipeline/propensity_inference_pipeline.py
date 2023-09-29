import data_ingestion
import data_transformation
import model_inference


if __name__ == '__main__':
    ingestor = data_ingestion.DataIngestion()
    transformer = data_transformation.DataTransformation()
    predictor = model_inference.PropensityModelInference()
    
    # Extract
    ingestor.initiate_data_ingestion()

    # Transform
    transformer.initiate_data_transformation()
    
    # Inference 
    predictor.initiate_propensity_inference()