import sys, os

sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from src.data_gen import *
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
    error_probabilities = jnp.array(error_model.probability_distribution(error_probability))
    error_permutations = jnp.array([[
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
    errors_A = sample_errors(key, code.size, error_probabilities, error_permutations)
    errors_B = sample_errors(key, code.size, error_probabilities)
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
    error_probabilities = jnp.array(error_model.probability_distribution(error_probability))
    error_permutations = jnp.array([[
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
    errors = sample_error_batch(key, batch_size, code.size, error_probabilities, error_permutations)
    # Assert
    assert errors.shape == (batch_size, 2*code.n_k_d[0])

def test_sample_deformation():
    # Arrange
    deformation_probabilities = jnp.array([
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [
            [0.2, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.1, 0.2, 0.2, 0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.2, 0.1, 0.2]
        ]
    ])
    key = random.key(0)
    # Act
    deformation = sample_deformation(key, deformation_probabilities)
    # Assert
    assert all(deformation[:6] == jnp.array([0,1,2,3,4,5])) # The first two rows are deterministic and independent of the key
    assert deformation.shape == (9,)

def test_sample_deformation_batch():
    # Arrange
    deformation_probabilities = jnp.array([
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [
            [0.2, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.1, 0.2, 0.2, 0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.2, 0.1, 0.2]
        ]
    ])
    key = random.key(0)
    batch_size = 5
    # Act
    deformations = sample_deformation_batch(key, batch_size, deformation_probabilities)
    # Assert
    assert all((deformations[:, :6] == jnp.array([jnp.array([0,1,2,3,4,5])]*batch_size)).flatten()) # The first two rows are deterministic and independent of the key
    assert deformations.shape == (batch_size, 9)

def test_transform_code_stabilizers():
    # Arrange
    class test_code:
        # Mimics the minimal attributes of a QEC code needed for this test
        stabilizers = jnp.array([
            [0,0,0,0,0,0,0,0,0,0,0,0], # I
            [1,1,1,1,1,1,0,0,0,0,0,0], # X
            [1,1,1,1,1,1,1,1,1,1,1,1], # Y
            [0,0,0,0,0,0,1,1,1,1,1,1], # Z
        ])
        logicals = stabilizers.copy()
        n_k_d = (6,0,0)
    code = test_code()
    deformation = jnp.array([0,1,2,3,4,5])
    expected_stabilizers = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    ])
    # Act
    transformed_stabilizers, transformed_logicals = transform_code_stabilizers(code, deformation)
    # Assert
    assert jnp.array_equal(transformed_stabilizers, expected_stabilizers)
    assert jnp.array_equal(transformed_logicals, expected_stabilizers)
    # For visual inspection (More human readable format, first row marks Xs, second row marks Zs)
    # print('I')
    # print(code.stabilizers[0].reshape(2,6))
    # print(transformed_stabilizers[0].reshape(2,6))
    # print('X')
    # print(code.stabilizers[1].reshape(2,6))
    # print(transformed_stabilizers[1].reshape(2,6))
    # print('Y')
    # print(code.stabilizers[2].reshape(2,6))
    # print(transformed_stabilizers[2].reshape(2,6))
    # print('Z')
    # print(code.stabilizers[3].reshape(2,6))
    # print(transformed_stabilizers[3].reshape(2,6))

def test_error_permutations_from_deformation():
    # Arrange
    deformation = jnp.array([
        [0,0,0],
        [0,1,2],
        [3,4,5]
    ])
    # Act
    error_permutations = error_permutations_from_deformation(deformation)
    # Assert
    expected_permutations = jnp.array([
        [[0,1,2,3],[0,1,2,3],[0,1,2,3]],
        [[0,1,2,3],[0,2,1,3],[0,1,3,2]],
        [[0,3,2,1],[0,2,3,1],[0,3,1,2]]
    ])
    assert jnp.array_equal(error_permutations, expected_permutations)

def test_syndrome_to_image_mapper():
    # Arrange
    code = RotatedPlanarCode(3,3)
    syndrome = jnp.array([1,2,3,4,5,6,7,8]) # In practice this is a binary array but for testing we use integers to see where they land in the image
    expected_image = jnp.array([[
        [0, 0, 1, 0],
        [2, 3, 4, 0],
        [0, 5, 6, 7],
        [0, 8, 0, 0],
    ]])
    # Act
    mapper = syndrome_to_image_mapper(code)
    image = mapper(syndrome)
    # Assert
    assert jnp.array_equal(image, expected_image)

def test_deformation_to_image_mapper():
    # Arrange
    code = RotatedPlanarCode(3,3)
    deformation = jnp.array([
        0,1,2,
        3,4,5,
        0,0,0
    ])
    expected_image = jnp.array([[
        # Deformation idx 0 (No deformation)
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 1]],
        # Deformation idx 1 (X-Y)
       [[0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]],
         # Deformation idx 2 (Y-Z)
       [[0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]],
        # Deformation idx 3 (X-Z)
       [[0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]],
        # Deformation idx 4 (X-Z-Y)
       [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]],
        # Deformation idx 5 (Y-Z-X)
       [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]]])
    # Act
    mapper = deformation_to_image_mapper(code)
    image = mapper(deformation)
    # Assert
    assert jnp.array_equal(image, expected_image)

if __name__ == "__main__":
    test_sample_errors()
    test_sample_error_batch()
    test_sample_deformation()
    test_sample_deformation_batch()
    test_transform_code_stabilizers()
    test_error_permutations_from_deformation()
    test_syndrome_to_image_mapper()
    test_deformation_to_image_mapper()
    print("All tests passed.")