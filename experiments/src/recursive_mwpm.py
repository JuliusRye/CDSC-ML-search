from qecsim.models.rotatedplanar import RotatedPlanarCode
from pymatching import Matching
from qecsim.models.generic import SimpleErrorModel
import jax.numpy as jnp

def recursive_mwpm(
    code: RotatedPlanarCode, 
    syndrome: jnp.ndarray, 
    noise_model: SimpleErrorModel,
    error_probability: float,
    noise_permutations: jnp.ndarray = None,
    iteration_limit=10,
    verbose=False,
) -> jnp.ndarray:
    """
    An implementation of the recursive MWPM (recMWPM) from 
    [this paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.022401), 
    based on the MWPM decoder implementation (`pymatching.Matching`) from the PyMatching package.

    Args:
        code (RotatedPlanarCode): The quantum error-correcting code.
        syndrome (jnp.ndarray): The syndrome to decode.
        noise_model (SimpleErrorModel): The error model to sample from.
        error_probability (float): The probability of an error occurring on a data qubit.
        noise_permutations (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit (used to implement the effect of a Clifford deformations on the code without altering the stabilizers).
        iteration_limit (int): The maximum number of iterations to perform.
        verbose (bool): Whether to print progress information.

    Returns:
        jnp.ndarray (shape (2*n_dat,)): The proposed recovery in binary symplectic form.
    """
    # Set default permutation if none provided
    if noise_permutations is None:
        noise_permutations = jnp.tile(jnp.array([0,1,2,3]), reps=(*code.size,1))
    else:
        assert code.size == noise_permutations.shape[:-1], "Permutation shape does not match code size"

    # Split stabilizers into X and Z parts
    n_stab, n_dat = code.stabilizers.shape
    stabilizer_x = code.stabilizers[n_stab//2:,:n_dat//2]
    stabilizer_z = code.stabilizers[:n_stab//2,n_dat//2:]

    # Calculate error weights
    probabilities = jnp.array(noise_model.probability_distribution(error_probability))
    deformed_probabilities = probabilities[noise_permutations].reshape((code.n_k_d[0], 4))
    wx = -jnp.log(deformed_probabilities[:,1])
    wy = -jnp.log(deformed_probabilities[:,2])
    wz = -jnp.log(deformed_probabilities[:,3])

    # Initial guess: no error
    recovery_x = jnp.zeros(n_dat//2, dtype=jnp.int32)
    recovery_z = jnp.zeros(n_dat//2, dtype=jnp.int32)
    recovery_weight = jnp.inf

    for i in range(iteration_limit):
        # Find lowest weight recovery for the X sub-graph
        decoder_x_plaquettes = Matching(
            graph=stabilizer_x,
            # Adjust weights based on current Z recovery
            weights=jnp.where(recovery_z == 1, wz-wy, wx),
        )
        recovery_x_new = decoder_x_plaquettes.decode(syndrome[n_stab//2:])

        # Find lowest weight recovery for the Z sub-graph
        decoder_z_plaquettes = Matching(
            graph=stabilizer_z,
            # Adjust weights based on current X recovery
            weights=jnp.where(recovery_x == 1, wy-wx, wz)
        )
        recovery_z_new = decoder_z_plaquettes.decode(syndrome[:n_stab//2:])

        # Calculate weight of the new recovery
        recovery_weight_new = sum([
            wx[jnp.where(jnp.logical_and(recovery_x_new==1, recovery_z_new==0))].sum(),
            wy[jnp.where(jnp.logical_and(recovery_x_new==1, recovery_z_new==1))].sum(),
            wz[jnp.where(jnp.logical_and(recovery_x_new==0, recovery_z_new==1))].sum(),
        ])

        if recovery_weight_new > recovery_weight:
            # If the weight increased, revert to previous and stop
            break
        else:
            recovery_weight = recovery_weight_new
        
        # If the recovery doesn't change, we are done
        if jnp.all(recovery_x_new == recovery_x) and jnp.all(recovery_z_new == recovery_z):
            break

        # Otherwise, update and repeat
        recovery_x = recovery_x_new
        recovery_z = recovery_z_new

        if verbose:
            print(f"Iteration {i+1} of {iteration_limit} with weight {recovery_weight}", end='\r')

    recovery = jnp.append(recovery_x, recovery_z)
    return recovery

def recursive_mwpm_batch(
    code: RotatedPlanarCode, 
    syndromes: jnp.ndarray, 
    noise_model: SimpleErrorModel,
    error_probability: float,
    noise_permutations: jnp.ndarray = None,
    iteration_limit=10,
) -> jnp.ndarray:
    """
    NOTE: This function is not yet optimized for performance and currently uses a simple for loop to apply the decoder to each syndrome in the batch.

    Apply the recursive MWPM decoder to a batch of syndromes in parallel.

    Args:
        syndromes (jnp.ndarray of shape (batch_size, n_stabilizers)): The syndromes to decode.
        code (RotatedPlanarCode): The quantum error-correcting code.
        noise_model (SimpleErrorModel): The error model to sample from.
        error_probability (float): The probability of an error occurring on a data qubit.
        noise_permutations (jnp.ndarray of shape (code.size, 4)): Optional permutation of the error probabilities for each qubit (used to implement the effect of a Clifford deformations on the code without altering the stabilizers).
        iteration_limit (int): The maximum number of iterations to perform.

    Returns:
        jnp.ndarray (shape (batch_size, 2*n_dat)): The proposed recoveries in binary symplectic form.
    """
    decoder = lambda syndrome: recursive_mwpm(code, syndrome, noise_model, error_probability, noise_permutations, iteration_limit)
    recoveries = jnp.zeros((syndromes.shape[0], 2*code.n_k_d[0]), dtype=jnp.int32)
    for i, syndrome in enumerate(syndromes):
        recovery = decoder(syndrome)
        recoveries = recoveries.at[i].set(recovery)
    return recoveries