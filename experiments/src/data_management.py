import sys, os
import json
import jax.numpy as jnp
from src.neural_network import mCNNDecoder, CNNDecoder
from qecsim.models.generic import BiasedDepolarizingErrorModel


def save_params(
    file_name: str,
    params: dict | list | tuple | jnp.ndarray | int | float,
):
    """
    Saves the neural network parameter object to a json file (replacing the jnp.ndarrays with lists).

    Args:
        file_name (str): The name of the JSON file to save the parameters to.
        params (dict | list | tuple | jnp.ndarray | int | float): The parameters to save.
    """
    def jsonify(obj: dict | list | tuple | jnp.ndarray | int | float):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(jsonify(v) for v in obj)
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, str):
            return obj
        raise NotImplementedError(
            f"Handling of type {type(obj)} has not been implemented")
    with open(file_name, 'w') as file:
        json.dump(jsonify(params), file, indent=4)


def load_params(
    file_name: str,
) -> dict:
    """
    Loads the neural network parameter object from a JSON file.

    Args:
        file_name (str): The name of the JSON file to load the parameters from.
    
    Returns:
        dict: The loaded parameters.
    """
    def de_jsonify(
        obj: dict | list | tuple | jnp.ndarray | int | float,
    ):
        if isinstance(obj, dict):
            return {k: de_jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            try:
                return jnp.array(obj)
            except (TypeError, ValueError):
                return [de_jsonify(v) for v in obj]
        if isinstance(obj, int) or isinstance(obj, float):
            return obj
        raise NotImplementedError(
            f"Handling of type {type(obj)} has not been implemented")
    with open(file_name, 'r') as file:
        data = json.load(file)
    return de_jsonify(data)


def load_model(name: str):
    """
    Load a trained model from the results directory.
    
    Args:
        name (str): The name of the model to load.
    
    Returns:
        tuple: A tuple containing:
            - model: The loaded model.
            - model_params: The parameters of the loaded model.
            - prefered_error_probabilities: The preferred error probabilities used during training.
    """
    path = f"results/{name}"
    if not os.path.exists(path):
        print(path)
        raise ValueError("Model not found")

    with open(f"{path}/settings.json", "r") as f:
        settings = json.load(f)
    with open(f"experiments/training_configs/{settings['<training_config>']}.json") as f:
        training_config = json.load(f)
    with open(f"{path}/nn_architecture.json", "r") as f:
        model_class, nn_architecture = json.load(f)
    model_params = load_params(f"{path}/model_params.json")

    match model_class:
        case "mCNNDecoder":
            model = mCNNDecoder(**nn_architecture)
        case "CNNDecoder":
            model = CNNDecoder(**nn_architecture)
        case _:
            raise ValueError(f"Unknown model class: {model_class}")

    prefered_error_probabilities = jnp.array(BiasedDepolarizingErrorModel(
        bias=training_config["ERROR_BIAS"],
        axis="Z"
    ).probability_distribution(training_config["ERROR_PROBABILITY"]))

    return model, model_params, prefered_error_probabilities