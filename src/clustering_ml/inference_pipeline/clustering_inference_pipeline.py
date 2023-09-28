import model_inference


if __name__ == '__main__':
    predictor = model_inference.ClusterModelInference()
    
    # Inference 
    predictor.initiate_cluster_inference()