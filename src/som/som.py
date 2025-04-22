## Kohonen SOM after modularization
import numpy as np

class SelfOrganisingMap():
    def __init__(self, width: int,
                 height: int,
                 input_dimension: int,
                 alpha: float = 0.1 ,
                 sigma: float =None):
        """Initializes the Self Organising Map
        The weights are initialized randomly between 0 and 1. The map is a 2D grid of neurons.


        Args:
            width (int): Width of the map
            height (int): Height of the map
            input_dimension (int): dimension of the data
            alpha (float, optional): The learning rate of the SOM . Defaults to 0.1.
            sigma (float, optional): The radius of the neighbourhood for the neurons. Defaults to max(width , height)/2.
            
        example usage:
            ```python
            som = SelfOrganisingMap(width= 10, height= 10, input_dimension= 3)
            ```
        """
        
        self.width = width
        self.height = height
        self.input_dimension = input_dimension
        self.alpha = alpha
        self.sigma = sigma or max(width, height) / 2
        self.weights = np.random.random((width, height, input_dimension))

    def find_bmu(self, vector: np.ndarray) -> tuple:
        """Find the best matching unit (BMU) for a given input vector.
        The BMU is the neuron whose weights are closest to the input vector in terms of Euclidean distance.

        Args:
            vector (np.ndarray): The input vector for which to find the BMU.

        Returns:
            tuple: returns the coordinates of the BMU in the 2D grid.
        The coordinates are in the form (x, y), where x is the width and y is the height of the map.
        
        example usage: 
            ```python
            som = SelfOrganisingMap(width = 10, height = 10, input_dimension= 3)
            bmu = som.find_bmu(vector)
            ```
        """
        distances = np.linalg.norm(self.weights - vector, axis=2)
        return np.unravel_index(np.argmin(distances), (self.width, self.height))

    def update_weights(self, vector: np.ndarray, bmu :tuple , t : int, max_iter: int) -> np.ndarray:
        """Updates the weights of the neurons in the map based on the input vector and the BMU.
        The weights are updated using a Gaussian neighborhood function and a learning rate that decays over time.
        The neighborhood function is defined by the distance between the BMU and the other neurons in the map.
        
        
        Args:
            vector (np.ndarray): The input vector used to update the weights. 
            bmu (tuple): The coordinates of the BMU on the map
            t (int): The current time step in the training process
            max_iter (int): The maximum number of iterations for the training process

        Returns:
            np.ndarray: Returns the updated weights of the neurons on the map. 
            
        example usage:
            ```python
            som = SelfOrganisingMap(width=10, height=10, input_dimension=3)
            som.update_weights(vector, bmu, t, max_iter)
            ```
        """
        lambda_val = max_iter / np.log(self.sigma)
        sigma_decay = self.sigma * np.exp(-t / lambda_val)
        alpha_decay = self.alpha * np.exp(-t / lambda_val)
        x, y = np.indices((self.width, self.height))
        distance_grid = (x - bmu[0])**2 + (y - bmu[1])**2
        theta = np.exp(-distance_grid / (2 * sigma_decay ** 2))[:, :, np.newaxis]
        self.weights += alpha_decay * theta * (vector - self.weights)
 
    def train(self, data: np.ndarray, num_iterations : int) -> np.ndarray:
        """Trains the self organising map based on the input data and number of iterations 

        Args:
            data (np.ndarray): The input data used to train the SOM map.
            num_iterations (int): The number of iterations for the training the map

        Returns:
            np.ndarray: Returns the trained weights of the neurons for the given metrics.
            
        Example usage:
            ```python
            som = SelfOrganisingMap(width=10, height=10, input_dimension=3)
            som.train(np.random.random((10, 3)),100)
            ```
        """
        for t in range(num_iterations):
            for vector in data:
                bmu = self.find_bmu(vector)
                self.update_weights(vector, bmu, t, num_iterations)
        
        return self.weights


