import sys, os

import jax
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from src.data_gen import errorpermutations_from_deformation, sample_errors, sample_error_batch
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
    errorpermutations = jnp.array([[
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
    errors_A = sample_errors(key, code, error_model, error_probability, errorpermutations)
    errors_B = sample_errors(key, code, error_model, error_probability)
    # Assert
    assert errors_A.shape == (2*code.n_k_d[0],)
    assert errors_B.shape == (2*code.n_k_d[0],)

def test_sample_error_batch():
    # Arrange
    key = random.key(0)
    batch_size = 5
    code = RotatedPlanarCode(3, 3)
    error_model = BiasedDepolarizingErrorModel(bias=500, axis='Z')
    error_probability = 0.1
    errorpermutations = jnp.array([[
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
    errors = sample_error_batch(key, batch_size, code, error_model, error_probability, errorpermutations)
    # Assert
    assert errors.shape == (batch_size, 2*code.n_k_d[0])

def test_errorpermutations_from_deformation():
    # Arrange
    deformation = jnp.array([
        [0,0,0],
        [0,1,2],
        [3,4,5]
    ])
    # Act
    errorpermutations = errorpermutations_from_deformation(deformation)
    # Assert
    expected_permutations = jnp.array([
        [[0,1,2,3],[0,1,2,3],[0,1,2,3]],
        [[0,1,2,3],[0,2,1,3],[0,1,3,2]],
        [[0,3,2,1],[0,2,3,1],[0,3,1,2]]
    ])
    assert jnp.array_equal(errorpermutations, expected_permutations)


if __name__ == "__main__":
    test_sample_errors()
    test_sample_error_batch()
    test_errorpermutations_from_deformation()
    print("All tests passed.")