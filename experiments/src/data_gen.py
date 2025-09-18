from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.generic import SimpleErrorModel
import jax.numpy as jnp
from jax import random, vmap


def sample_errors(
    key,
    code: RotatedPlanarCode,
    error_model: SimpleErrorModel,
    error_probability: float,
    noise_permutations: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Sample errors from the given error model for the given code.

    Args:
        key: JAX random key.
        code (RotatedPlanarCode): The quantum error-correcting code.
        error_model (SimpleErrorModel): The error model to sample from.
        error_probability (float): The probability of an error occurring on each qubit.
        noise_permutation (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit.
    
    Returns:
        jnp.ndarray: The sampled error in binary symplectic form.
    """
    # Set default permutation if none provided
    if noise_permutations is None:
        noise_permutations = jnp.tile(jnp.array([0,1,2,3]), reps=(*code.size,1))
    else:
        assert code.size == noise_permutations.shape[:-1], "Permutation shape does not match code size"
    # Sample errors according to the error model
    rv = random.uniform(key, shape=code.size)[:,:,None]
    probabilities = jnp.array(error_model.probability_distribution(error_probability))
    probabilities = probabilities[noise_permutations] # Permute probabilities if needed
    cumulative_probabilities = jnp.cumsum(probabilities, axis=-1)
    error_idxs = (cumulative_probabilities > rv).argmax(axis=-1).flatten() # 0: I, 1: X, 2: Y, 3: Z
    # Convert to binary symplectic form
    pauli_z_error = jnp.logical_or(error_idxs == 1, error_idxs == 2).astype(int)
    pauli_x_error = jnp.logical_or(error_idxs == 3, error_idxs == 2).astype(int)
    bsr = jnp.append(pauli_x_error, pauli_z_error)
    return bsr


def sample_error_batch(
    key, 
    batch_size: int, 
    code: RotatedPlanarCode, 
    noise_model: SimpleErrorModel, 
    error_probability: float, 
    noise_permutations: jnp.ndarray = None
):
    """
    Sample a batch of errors from the given error model for the given code.
    
    Args:
        key: JAX random key.
        batch_size (int): The number of errors to sample.
        code (RotatedPlanarCode): The quantum error-correcting code.
        error_model (SimpleErrorModel): The error model to sample from.
        error_probability (float): The probability of an error occurring on each qubit.
        noise_permutation (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit.
    
    Returns:
        jnp.ndarray: The sampled error in binary symplectic form.
    """
    keys = random.split(key, num=batch_size)
    errors = vmap(
        sample_errors,
        in_axes=(0, None, None, None, None),
        out_axes=0
    )(keys, code, noise_model, error_probability, noise_permutations)
    return errors


def noise_permutations_from_deformation(deformation: jnp.ndarray) -> jnp.ndarray:
    r"""
    Translates a deformation on the QECC into a corresponding permutation of the noise model (acting on an un-deformed version of the QECC).

    Args:
        deformation (jnp.ndarray): An array of the deformation (by index) for each data qubit.

    Returns:
        jnp.ndarray: An array of shape (`deformation.shape`, 4) representing the
    """
    # Map from deformation index to noise model permutation using the inverse transformation
    transformation_map = jnp.array([
    #    I  X  Y  Z
        [0, 1, 2, 3],  # I
        [0, 2, 1, 3],  # X-Y
        [0, 1, 3, 2],  # Y-Z
        [0, 3, 2, 1],  # X-Z
        [0, 2, 3, 1],  # X-Z-Y
        [0, 3, 1, 2],  # Y-Z-X
    ])
    return transformation_map[deformation]