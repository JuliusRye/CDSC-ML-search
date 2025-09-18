import sys, os
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from pymatching import Matching
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarPauli
from qecsim.models.generic import BiasedDepolarizingErrorModel
import jax.numpy as jnp
from jax import random
from src.data_gen import sample_errors, sample_error_batch
from src.recursive_mwpm import recursive_mwpm, recursive_mwpm_batch


def test_recursive_mwpm():
    # Arrange
    key = random.key(0)
    code = RotatedPlanarCode(3,3)
    noise_model = BiasedDepolarizingErrorModel(bias=10.0, axis='Y')
    noise_permutations = jnp.array([
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
    ])
    error_probability = 0.1
    # Act
    error = sample_errors(key, code, noise_model, error_probability, noise_permutations)
    syndrome = code.stabilizers @ error % 2
    recovery = recursive_mwpm(code, syndrome, noise_model, error_probability, noise_permutations, iteration_limit=10)
    recovery_syndrome = code.stabilizers @ recovery % 2
    # Assert
    assert error.shape == recovery.shape, "Error and recovery shapes do not match"
    assert jnp.array_equal(syndrome, recovery_syndrome), "Recovery does not match syndrome"

def test_recursive_mwpm_batch():
    # Arrange
    key = random.key(0)
    code = RotatedPlanarCode(3,3)
    noise_model = BiasedDepolarizingErrorModel(bias=10.0, axis='Y')
    noise_permutations = jnp.array([
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
    ])
    error_probability = 0.1
    batch_size = 5
    # Act
    errors = sample_error_batch(key, batch_size, code, noise_model, error_probability, noise_permutations)
    syndrome = jnp.array([code.stabilizers @ error % 2 for error in errors])
    recoveries = recursive_mwpm_batch(code, syndrome, noise_model, error_probability, noise_permutations, iteration_limit=10)
    recoveries_syndrome = jnp.array([code.stabilizers @ rec % 2 for rec in recoveries])
    # Assert
    assert errors.shape == recoveries.shape, "Error and recovery shapes do not match"
    assert jnp.array_equal(syndrome, recoveries_syndrome), "Recovery does not match syndrome"


if __name__ == "__main__":
    test_recursive_mwpm()
    test_recursive_mwpm_batch()
    print("All tests passed!")