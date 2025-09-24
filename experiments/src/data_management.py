import sys, os
import json
import jax.numpy as jnp
from src.neural_network import mCNNDecoder, CNNDecoder
from qecsim.models.generic import BiasedDepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode


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
            - code: The quantum error-correcting code.
            - trained_using: A dictionary containing the error "error_probabilities" and "deformation" the model was trained with.
    """
    path = f"results/{name}"
    config_path = "experiments/training_configs"
    if not os.path.exists(path):
        path = f"../results/{name}" # For use in Jupyter notebooks
        config_path = "training_configs"
    if not os.path.exists(path):
        raise ValueError("Model not found", path)

    with open(f"{path}/settings.json", "r") as f:
        settings = json.load(f)
    with open(f"{config_path}/{settings['<training_config>']}.json") as f:
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
    
    L = int(settings["<code_distance>"])
    code = RotatedPlanarCode(L, L)

    trained_using = {}
    trained_using["error_probabilities"] = jnp.array(BiasedDepolarizingErrorModel(
        bias=training_config["ERROR_BIAS"],
        axis="Z"
    ).probability_distribution(training_config["ERROR_PROBABILITY"]))
    deformation = deformation_from_name(code, settings["<deformation_name>"])
    if isinstance(deformation, jnp.ndarray):
        trained_using["deformation"] = deformation
    else:
        trained_using["deformation"] = None

    return model, model_params, code, trained_using

def deformation_from_name(code: RotatedPlanarCode, deformation_name: str):
    match deformation_name:
        case "Generalized":
            return "Generalized"
        case "Guided":
            return "Guided"
        case "CSS":
            return jnp.zeros(code.size[0]*code.size[1], dtype=jnp.int32)
        case "XZZX":
            return jnp.zeros(code.size[0]*code.size[1], dtype=jnp.int32).at[::2].set(3)
        case "XY":
            return jnp.zeros(code.size[0]*code.size[1], dtype=jnp.int32).at[:].set(2)
        case "C1":
            return jnp.zeros((code.size[0], code.size[1]), dtype=jnp.int32).at[1::2, ::2].set(3).at[::2,::2].set(2).at[1::2,1::2].set(2).flatten()
        case _:
            if all(char in "012345" for char in deformation_name) and len(deformation_name) == code.size[0]*code.size[1]:
                return jnp.array([int(char) for char in deformation_name], dtype=jnp.int32)
            else:
                raise ValueError(f"Unknown deformation_name: {deformation_name}")