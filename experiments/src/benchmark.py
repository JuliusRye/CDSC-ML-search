import jax.numpy as jnp
from src.data_management import load_model
from src.data_gen import data_batch, deformation_to_image_mapper
from src.neural_network import mCNNDecoder, CNNDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode
from icecream import ic


def benchmark_nn_decoder(
    key,
    batch_size: int,
    model_name: str,
    error_probabilities: jnp.ndarray = None,
    deformation: jnp.ndarray = None,
    with_histogram_2d=False,
    verbose=False
):
    """
    Benchmark a neural network decoder on a given code, error model and deformation.

    Args:
        key: JAX random key.
        batch_size (int): Number of samples to use for benchmarking.
        model_name (str): Name of the model to load.
        code (RotatedPlanarCode): The quantum error-correcting code.
        error_probabilities (jnp.ndarray | None): Array of error probabilities for each Pauli error or None for the models trained with error_probabilities.
        deformation (jnp.ndarray | None): Deformation to apply to the code or None for the deformation the model was trained with.
        with_histogram_2d (bool): Whether to return a 2D histogram of results comparing model predictions with true error logicals.
        verbose (bool): Whether to print detailed information during benchmarking.
    
    Returns:
        tuple: tuple containing:
            - logical_error_rate (float): The logical error rate of the model.
            - hist2d (jnp.ndarray, optional): 2D histogram of model predictions if with_histogram_2d is True.
    """
    model, model_params, code, trained_using = load_model(model_name)
    if error_probabilities is None:
        error_probabilities = trained_using["error_probabilities"]
        if verbose:
            print("Using the error probabilities the model was trained with:", error_probabilities)
    if deformation is None:
        deformation = trained_using["deformation"]
        if deformation is None:
            raise ValueError("The model was not trained on a specific deformation, please provide one for this benchmark.")
        if verbose:
            print("Using the deformation the model was trained with:\n", deformation.reshape(code.size))
    error_syndromes_img, error_logicals = data_batch(key, batch_size, code, error_probabilities, deformation.reshape(code.size), as_images=True)
    deformation_img: jnp.ndarray = deformation_to_image_mapper(code)(deformation)[None,:,:,:]

    if isinstance(model, mCNNDecoder):
        deformation_img = deformation_to_image_mapper(code)(deformation)[None,:,:,:]
        model_prediction = (model.apply_batch(
            model_params, 
            error_syndromes_img.astype(jnp.float32), 
            deformation_img.astype(jnp.float32)
        ) > 0).astype(jnp.int32)
    elif isinstance(model, CNNDecoder):
        model_prediction = (model.apply_batch(
            model_params, 
            error_syndromes_img.astype(jnp.float32)
        ) > 0).astype(jnp.int32)
    else:
        raise ValueError("Unknown model type", type(model))
    
    logical_error_rate = jnp.mean(jnp.any(model_prediction != error_logicals, axis=1))

    if not with_histogram_2d:
        return logical_error_rate

    i = 2 * error_logicals[:, 0] + error_logicals[:, 1]
    j = 2 * model_prediction[:, 0] + model_prediction[:, 1]
    hist2d = jnp.zeros((4,4), dtype=jnp.int32).at[i, j].add(1)

    return logical_error_rate, hist2d
