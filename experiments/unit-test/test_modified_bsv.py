import sys, os
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarPauli, RotatedPlanarRMPSDecoder
from src.ModifiedRotatedPlanarRMPSDecoder import ModifiedRotatedPlanarRMPSDecoder
from src.data_gen import noise_permutations_from_deformation, sample_errors
from qecsim.models.generic import BiasedDepolarizingErrorModel
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random


def test_modified_bsv():
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


if __name__ == "__main__":
    test_modified_bsv()