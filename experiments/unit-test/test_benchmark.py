import sys, os

sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from src.benchmark import benchmark_nn_decoder
from qecsim.models.rotatedplanar import RotatedPlanarCode
import jax.numpy as jnp
from jax import random
from icecream import ic


def test_benchmark_nn_decoder():
    # Arrange
    model_name = "test"
    error_probabilities = jnp.array([8.9999998e-01, 9.9800396e-05, 9.9800396e-05, 9.9800400e-02])
    deformation = jnp.array([2,0,2,3,2,3,2,0,2])  # C1 for 3x3
    batch_size = 1_000
    key = random.key(0)
    # Act
    ler_a, hist2d_a = benchmark_nn_decoder(
        key,
        batch_size,
        model_name,
        error_probabilities,
        deformation,
        with_histogram_2d=True
    )
    ler_b, hist2d_b = benchmark_nn_decoder(
        key,
        batch_size,
        model_name,
        error_probabilities,
        deformation,
        with_histogram_2d=True
    )
    # Assert
    assert all([ler_a >= 0, ler_a <= 1]), "Logical error rate is out of bounds"
    assert hist2d_a.shape == (4, 4), "2D histogram shape is incorrect"
    assert jnp.isclose(hist2d_a.sum(), batch_size), "2D histogram sum does not equal batch size"
    assert jnp.array_equal(hist2d_a, hist2d_b), "Results are not deterministic"
    assert jnp.isclose(ler_a, ler_b), "Results are not deterministic"

if __name__ == "__main__":
    test_benchmark_nn_decoder()
    print("All tests passed!")
