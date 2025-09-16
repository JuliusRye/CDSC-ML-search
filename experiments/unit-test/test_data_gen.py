import sys, os

import jax
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from src.data_gen import noise_permutations_from_deformation, sample_errors
import jax.numpy as jnp
from jax import random
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.generic import BiasedDepolarizingErrorModel


def test_sample_errors():
    # Arrange
    key = random.key(0)
    code = RotatedPlanarCode(3, 3)
    error_model = BiasedDepolarizingErrorModel(bias=500, axis='Z')
    error_probability = 0.1
    noise_permutations = jnp.array([[
        [0, 1, 2, 3], # No deformation
        [0, 1, 2, 3], # No deformation
        [0, 1, 2, 3]],# No deformation

       [[0, 1, 2, 3], # No deformation
        [0, 2, 1, 3], # X-Y deformation
        [0, 1, 3, 2]],# Y-Z deformation

       [[0, 3, 2, 1], # X-Z deformation
        [0, 2, 3, 1], # X-Z-Y deformation
        [0, 3, 1, 2]] # Y-Z-X deformation
    ])
    # Act
    errors_A = sample_errors(key, code, error_model, error_probability, noise_permutations)
    errors_B = sample_errors(key, code, error_model, error_probability)
    # Assert
    print(errors_A.shape)
    assert errors_A.shape == (2*code.n_k_d[0],)
    assert errors_B.shape == (2*code.n_k_d[0],)

def test_noise_permutations_from_deformation():
    # Arrange
    deformation = jnp.array([
        [0,0,0],
        [0,1,2],
        [3,4,5]
    ])
    # Act
    noise_permutations = noise_permutations_from_deformation(deformation)
    # Assert
    expected_permutations = jnp.array([
        [[0,1,2,3],[0,1,2,3],[0,1,2,3]],
        [[0,1,2,3],[0,2,1,3],[0,1,3,2]],
        [[0,3,2,1],[0,2,3,1],[0,3,1,2]]
    ])
    assert jnp.array_equal(noise_permutations, expected_permutations)


if __name__ == "__main__":
    test_sample_errors()
    print("All tests passed.")