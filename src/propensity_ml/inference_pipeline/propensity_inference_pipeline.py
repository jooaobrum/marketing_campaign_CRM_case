import model_inference


if __name__ == '__main__':
    predictor = model_inference.PropensityModelInference()
    
    # Inference 
    predictor.initiate_propensity_inference()