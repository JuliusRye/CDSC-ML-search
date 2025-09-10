import sys, os
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

import pytest
import jax.numpy as jnp
from jax import random
from src.quantum_error_correction_code import QEC, SurfaceCode

def test_surface_code():
    # Arrange
    L = 5
    # Act
    qec = SurfaceCode(L=L)
    # Assert
    assert qec.num_syndrome_qubits == 24
    assert qec.num_data_qubits == 25
    assert qec.lx_original.shape == (2, 25)
    assert qec.lz_original.shape == (2, 25)
    assert qec.hx_original.shape == (24, 25)
    assert qec.hz_original.shape == (24, 25)

def test_random_deformation():
    # Arrange
    qec = SurfaceCode(L=3)
    key = random.key(0)
    probs = jnp.ones(shape=(qec.num_data_qubits, 6)) * jnp.array([0.0, 0.8, 0.0, 0.2, 0.0, 0.0])[None, :]
    # Act
    deformation_1, new_key = qec.random_deformation(key, probs)
    deformation_2, new_key = qec.random_deformation(key, probs)
    # Assert
    assert deformation_1.shape[0] == qec.num_data_qubits
    assert jnp.all(deformation_1 == deformation_2) # check determinism
    assert jnp.all(jnp.isin(deformation_1, jnp.array([1,3])))
    assert new_key != key

def test_deformation_parity_info():
    # Arrange
    qec = SurfaceCode(L=5)
    deformation = jnp.array([0,3,1,3,2,1,0,2,0,1,2,3,1,2,3,0,1,2,3,0,1,2,3,0,1])  # Example deformation
    # Act
    hx, hz, lx, lz = qec.deformation_parity_info(deformation)
    # Assert
    assert hx.shape == (24, 25)
    assert hz.shape == (24, 25)
    assert lx.shape == (2, 25)
    assert lz.shape == (2, 25)

def test_error():
    # Arrange
    qec = SurfaceCode(L=5)
    key = random.key(0)
    px, py, pz = 0.1, 0.2, 0.3
    # Act
    errors = qec.error(key, [px, py, pz])
    # Assert
    assert errors.shape == (2, 25)

def test_syndrome():
    # Arrange
    qec = SurfaceCode(L=5)
    error = jnp.array([
        [1,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0],  # X errors
        [0,1,0,0,0, 0,0,0,1,0, 0,0,0,0,0, 0,1,0,0,0, 0,0,0,0,1],  # Z errors
    ])
    # Act
    parity_info = qec.deformation_parity_info(jnp.zeros(qec.num_data_qubits, dtype=int))
    syndrome, logical_1 = qec.syndrome(error, parity_info)
    syndrome_img, logical_2 = qec.syndrome_img(error, parity_info)
    # Assert
    assert syndrome.shape == (24,)
    assert syndrome_img.shape == (6,6)
    assert logical_1.shape == (2,)
    assert jnp.all(logical_1 == logical_2)

def test_deformation_image():
    # Arrange
    qec = SurfaceCode(L=5)
    deformation = jnp.array([0,3,1,3,2,1,0,2,0,1,2,3,1,2,3,0,1,2,3,0,1,2,3,0,1])  # Example deformation
    # Act
    deformation_img = qec.deformation_image(deformation)
    # Assert
    assert deformation_img.shape == (6,5,5)

if __name__ == "__main__":
    test_random_deformation()
    # test_deformation_parity_info()