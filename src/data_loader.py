import numpy as np

def generate_random_data(num_samples: int, num_features: int = 3) -> np.ndarray:
    """
    Generates random data samples.
    
    Args: 
        num_samples (int): The number of samples to generate.
        num_features (int): The number of features for each sample. Default is 3.
    
    Returns:
        np.ndarray: A 2D array of shape (num_samples, num_features) containing random samples.
    """
    return np.random.rand(num_samples, num_features)
