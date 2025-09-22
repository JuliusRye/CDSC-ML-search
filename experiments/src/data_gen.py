from functools import partial
from qecsim.models.rotatedplanar import RotatedPlanarCode
import jax.numpy as jnp
from jax import random, vmap, jit

# Static variables
relevancy_tensor = jnp.array([
    # No error
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    # Pauli X error
    [[1, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 1],
     [0, 1, 0, 0, 1, 0],
     [0, 0, 0, 1, 0, 1]],
    # Pauli Z error
    [[1, 1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 1],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 1, 0, 0, 1]],
    # Pauli Y error
    [[1, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 1],
     [0, 0, 1, 0, 1, 0],
     [1, 0, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0],
     [0, 1, 0, 0, 0, 1]],
])

# Sampling functions
@partial(jit, static_argnames=("code_shape",))
def _sample_errors(
    key,
    code_shape: tuple[int, int],
    error_probabilities: jnp.ndarray,
    error_permutations: jnp.ndarray = None
) -> jnp.ndarray:
    # Set default permutation if none provided
    if error_permutations is None:
        error_permutations = jnp.tile(jnp.array([0,1,2,3]), reps=(*code_shape,1))
    # else:
    #     assert code_shape == error_permutations.shape[:-1], "Permutation shape does not match code size"
    # Sample errors according to the error model
    rv = random.uniform(key, shape=code_shape)[:,:,None]
    error_probabilities = error_probabilities[error_permutations] # Permute probabilities if needed
    cumulative_probabilities = jnp.cumsum(error_probabilities, axis=-1)
    error_idxs = (cumulative_probabilities > rv).argmax(axis=-1).flatten() # 0: I, 1: X, 2: Y, 3: Z
    # Convert to binary symplectic form
    pauli_z_error = jnp.logical_or(error_idxs == 1, error_idxs == 2).astype(int)
    pauli_x_error = jnp.logical_or(error_idxs == 3, error_idxs == 2).astype(int)
    bsr = jnp.append(pauli_x_error, pauli_z_error)
    return bsr

def sample_errors(
    key,
    code_shape: tuple[int, int],
    error_probabilities: jnp.ndarray,
    error_permutations: jnp.ndarray = None
) -> jnp.ndarray:
    """
    #### Jit optimized function!

    Sample errors from the given error model for the given code.

    Args:
        key: JAX random key.
        code_shape (tuple[int, int]): The shape of the quantum error-correcting code.
        error_probabilities (jnp.ndarray of shape (4,)): The probabilities of [I, X, Y, Z] errors on each qubit.
        error_permutations (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit.

    Returns:
        jnp.ndarray: The sampled error in binary symplectic form.
    """
    return _sample_errors(key, code_shape, error_probabilities, error_permutations)

@partial(jit, static_argnames=("batch_size", "code_shape",))
def _sample_error_batch(
    key, 
    batch_size: int, 
    code_shape: tuple[int, int],
    error_probabilities: jnp.ndarray,
    error_permutations: jnp.ndarray = None
):
    keys = random.split(key, num=batch_size)
    errors = vmap(
        sample_errors,
        in_axes=(0, None, None, None),
        out_axes=0
    )(keys, code_shape, error_probabilities, error_permutations)
    return errors

def sample_error_batch(
    key, 
    batch_size: int, 
    code_shape: tuple[int, int],
    error_probabilities: jnp.ndarray,
    error_permutations: jnp.ndarray = None
):
    """
    #### Jit optimized function!

    Sample a batch of errors from the given error model for the given code.
    
    Args:
        key: JAX random key.
        batch_size (int): The number of errors to sample.
        code (RotatedPlanarCode): The quantum error-correcting code.
        error_probabilities (jnp.ndarray of shape (4,)): The probabilities of [I, X, Y, Z] errors on each qubit.
        errorpermutation (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit.
    
    Returns:
        jnp.ndarray: The sampled error in binary symplectic form.
    """
    return _sample_error_batch(key, batch_size, code_shape, error_probabilities, error_permutations)

@jit
def _sample_deformation(
    key, 
    deformation_probabilities: jnp.ndarray
):
    # Cumulative probability
    cp = jnp.cumsum(deformation_probabilities.reshape(-1, 6).T, axis=0)
    # Random variable for sampling
    rv = random.uniform(key, shape=cp.shape[1])
    # Find the first index where the cumulative probability is greater than the random variable
    return (cp > rv).argmax(axis=0)

def sample_deformation(
    key, 
    deformation_probabilities: jnp.ndarray
):
    """
    #### Jit optimized function!

    Sample a deformation from the given deformation probabilities.
    
    Args:
        key: JAX random key.
        deformation_probabilities (jnp.ndarray of shape (deformation.shape, 6)): The probabilities of each deformation for each data qubit.

    Returns:
        jnp.ndarray: An array of shape (deformation.shape,) representing the sampled deformation.
    """
    return _sample_deformation(key, deformation_probabilities)

@partial(jit, static_argnames=("batch_size",))
def _sample_deformation_batch(
    key, 
    batch_size: int, 
    deformation_probabilities: jnp.ndarray
):
    return vmap(sample_deformation, in_axes=(0, None))(random.split(key, batch_size), deformation_probabilities)

def sample_deformation_batch(
    key, 
    batch_size: int, 
    deformation_probabilities: jnp.ndarray
):
    """
    #### Jit optimized function!

    Sample a batch of deformations from the given deformation probabilities.
    
    Args:
        key: JAX random key.
        batch_size (int): The number of deformations to sample.
        deformation_probabilities (jnp.ndarray of shape (deformation.shape, 6)): The probabilities of each deformation for each data qubit.

    Returns:
        jnp.ndarray: An array of shape (batch_size, deformation.shape) representing the sampled deformations.
    """
    return _sample_deformation_batch(key, batch_size, deformation_probabilities)

def data_batch(
    key,
    batch_size: int,
    code: RotatedPlanarCode,
    error_probabilities: jnp.ndarray,
    deformation: jnp.ndarray,
    as_images: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, any]:
    """
    Generate a batch of syndromes and logicals for the given code, error model, and deformation.

    Args:
        key: JAX random key.
        batch_size (int): The number of samples to generate.
        code (RotatedPlanarCode): The quantum error-correcting code.
        error_probabilities (jnp.ndarray of shape (4,)): The probabilities of [I, X, Y, Z] errors on each qubit.
        deformation (jnp.ndarray of shape (code.size,)): The deformation to apply to the code.
        as_images (bool): Whether to return syndromes as images or flat vectors.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - syndromes (jnp.ndarray): The syndromes of shape (batch_size, syndrome_size) or (batch_size, code.size+1, code.size+1, 1) if as_images is True.
            - logicals (jnp.ndarray): The logicals of shape (batch_size, 2).
    """
    error_permutation = error_permutations_from_deformation(deformation)
    deformed_errors = sample_error_batch(key, batch_size, code.size, error_probabilities, error_permutation)
    error_syndromes = vmap(lambda s,e: (s @ e) % 2, in_axes=(0, 0))(code.stabilizers, deformed_errors)
    error_logicals = vmap(lambda l,e: (l @ e) % 2, in_axes=(0, 0))(code.logicals, deformed_errors)
    if as_images:
        syndrome_mapper = syndrome_to_image_mapper(code)
        error_syndromes = vmap(syndrome_mapper)(error_syndromes)
    return error_syndromes, error_logicals

# Transformation functions
def transform_code_stabilizers(
    code: RotatedPlanarCode,
    deformation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Given a deformation and a quantum error correction code, return the transformed stabilizers and logicals.

    Args:
        code: A quantum error correction code (e.g., RotatedPlanarCode).
        deformation: A 2D array where each entry is an integer in [0, 5] representing the type of deformation applied to each qubit.
    
    Returns:
        A tuple containing:
            - stabilizers: The transformed stabilizers.
            - logicals: The transformed logicals.
    """

    # Ensure deformation is a flat array
    deformation = deformation.flatten()

    # Map of how each entry (hx_i,j, hz_i,j) in the parity check matrix transforms under each deformation
    transformation_map = jnp.array([
    #   X(1, 0) Z(0, 1) turn into:
        [[1, 0], [0, 1]],  # I
        [[1, 1], [0, 1]],  # X-Y
        [[1, 0], [1, 1]],  # Y-Z
        [[0, 1], [1, 0]],  # X-Z
        [[0, 1], [1, 1]],  # X-Z-Y
        [[1, 1], [1, 0]],  # X-Y-Z
    ]).transpose(0,2,1)

    # Number of data qubits and stabilizers
    n_dat = code.n_k_d[0]
    n_stab = code.stabilizers.shape[0]

    # Combine stabilizers and logicals for transformation
    A = jnp.append(code.stabilizers[:,:n_dat], code.logicals[:,:n_dat], axis=0)
    B = jnp.append(code.stabilizers[:,n_dat:], code.logicals[:,n_dat:], axis=0)

    # Apply transformation column-wise
    A_prime, B_prime = vmap(
        lambda A, B, Di: jnp.dot(Di, jnp.stack([A, B])) % 2,
        in_axes=(1, 1, 0),
        out_axes=2
    )(
        A,
        B,
        transformation_map[deformation]
    )

    # Split back into stabilizers and logicals
    stabilizers = jnp.append(A_prime[:n_stab, :], B_prime[:n_stab, :], axis=1)
    logicals    = jnp.append(A_prime[n_stab:, :], B_prime[n_stab:, :], axis=1)

    return stabilizers, logicals

def error_permutations_from_deformation(deformation: jnp.ndarray) -> jnp.ndarray:
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

def syndrome_to_image_mapper(
    code: RotatedPlanarCode
) -> callable:
    """
    Given an instance of the surface code, return a function that transforms a syndrome vector into an image.

    Args:
        code (RotatedPlanarCode): The quantum error-correcting code.
    
    Returns:
        fun_syndrome_to_img (callable): A function that takes a syndrome vector and returns an image representation of the syndrome (code.size, 1).
    """
    mask = jnp.array([[code.is_in_plaquette_bounds((i, j)) for i in range(-1,code.size[1])] for j in range(-1,code.size[0])])
    shape = tuple(s + 1 for s in code.size)

    def fun_syndrome_to_img(syndrome):
        return jnp.zeros(shape, dtype=jnp.int32).at[mask].set(syndrome)[:,:,None].transpose(2, 0, 1)

    return fun_syndrome_to_img

def deformation_to_image_mapper(
    code: RotatedPlanarCode,
) -> callable:
    """
    Given an instance of the surface code, return a function that transforms a deformation vector into an image.

    Args:
        code (RotatedPlanarCode): The quantum error-correcting code.
    
    Returns:
        fun_deformation_to_img (callable): A function that takes a deformation vector and returns an image representation of the deformation (code.size, 6).
    """

    # Done this way for consistency with syndrome_to_image_mapper
    def deformation_to_image(deformation):
        return jnp.eye(6, dtype=jnp.int32)[
            deformation.reshape(code.size)
        ].transpose(2, 0, 1)

    return deformation_to_image
