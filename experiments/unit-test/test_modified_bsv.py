import sys, os
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from qecsim.models.rotatedplanar import RotatedPlanarCode
from src.ModifiedRotatedPlanarRMPSDecoder import ModifiedRotatedPlanarRMPSDecoder
from src.data_gen import sample_errors, sample_error_batch
from qecsim.models.generic import BiasedDepolarizingErrorModel
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random


def test_modified_bsv():
    # Arrange
    key = random.key(0)
    code = RotatedPlanarCode(3,3)
    noise_model = BiasedDepolarizingErrorModel(bias=10.0, axis='Z')
    noise_permutations = jnp.array([
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
    ])
    error_probability = 0.1
    chi = 6
    # Act
    error = sample_errors(key, code, noise_model, error_probability, noise_permutations)
    syndrome = code.stabilizers @ error % 2
    recovery = ModifiedRotatedPlanarRMPSDecoder(chi).decode(
        code, syndrome, noise_model, error_probability, noise_permutations
    )
    recovery_syndrome = code.stabilizers @ recovery % 2
    # Assert
    assert error.shape == recovery.shape, "Error and recovery shapes do not match"
    assert jnp.array_equal(syndrome, recovery_syndrome), "Recovery does not match syndrome"

def test_modified_bsv_batch():
    # Arrange
    key = random.key(0)
    code = RotatedPlanarCode(3,3)
    noise_model = BiasedDepolarizingErrorModel(bias=10.0, axis='Z')
    noise_permutations = jnp.array([
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
        [[0,1,2,3], [0,1,2,3], [0,1,2,3]],
    ])
    error_probability = 0.1
    batch_size = 5
    chi = 6
    # Act
    errors = sample_error_batch(key, batch_size, code, noise_model, error_probability, noise_permutations)
    syndromes = jnp.array([code.stabilizers @ error % 2 for error in errors])
    recoveries = ModifiedRotatedPlanarRMPSDecoder(chi).decode_batch(
        code, syndromes, noise_model, error_probability, noise_permutations
    )
    recoveries_syndrome = jnp.array([code.stabilizers @ rec % 2 for rec in recoveries])
    # Assert
    assert errors.shape == recoveries.shape, "Error and recovery shapes do not match"
    assert jnp.array_equal(syndromes, recoveries_syndrome), "Recovery does not match syndrome"

if __name__ == "__main__":
    test_modified_bsv()
    test_modified_bsv_batch()
    print("All tests passed!")