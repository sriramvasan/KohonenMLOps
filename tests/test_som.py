import numpy as np
import pytest
from src.som.som import SelfOrganisingMap

@pytest.fixture
def som():
    return SelfOrganisingMap(width=10, height=10, input_dimension=3)

def test_initial_weights_shape(som):
    assert som.weights.shape == (10, 10, 3), "Initial weights shape is incorrect."

def test_bmu_index(som):
    input_vector = np.array([0.5, 0.5, 0.5])
    bmu_idx = som.find_bmu(input_vector)
    assert len(bmu_idx) == 2, "BMU index should have two coordinates."

def test_training_updates_weights(som):
    initial_weights = som.weights.copy()
    data = np.random.rand(5, 3)
    som.train(data, num_iterations=10)
    assert not np.array_equal(initial_weights, som.weights), "Weights should update after training."
