from data_loader import generate_random_data
from som.som import SelfOrganisingMap
from visualization import save_som_image
# Visualization
import os
output_dir = 'data/processed'

def main():
    # Parameters
    width, height = 10, 10
    input_dim = 3
    num_samples = 10
    num_iterations = 100

    # Data generation
    data = generate_random_data(num_samples, input_dim)

    # SOM initialization and training
    som = SelfOrganisingMap(width, height, input_dim)
    som.train(data, num_iterations)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'som_output{width}x{height}_{num_iterations}.png')
    save_som_image(som.weights,filepath, som, True, data)

if __name__ == "__main__":
    main()
