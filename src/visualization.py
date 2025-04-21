import matplotlib.pyplot as plt
import numpy as np
from som.som import SelfOrganisingMap

def save_som_image(weights: np.ndarray, filename: str, som_model : SelfOrganisingMap = None, show_bmu : bool= False, data: np.ndarray = None) -> None:
    """
    Saves the SOM weights as an image.
    
    Args:
        weights (np.ndarray): The weights of the SOM.
        filename (str): The filename to save the image.
        som_model (SelfOrganisingMap, optional): The SOM model.
        show_bmu (bool): Whether to show the best matching unit (BMU) on the image. Default is False.
    """
    plt.imshow(weights)
    plt.axis('off')
    
    if show_bmu:
        [plt.scatter(som_model.find_bmu(v) [0], som_model.find_bmu(v)[1], s =60 , edgecolors='w',c =np.zeros(0)) for v in data]  
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
