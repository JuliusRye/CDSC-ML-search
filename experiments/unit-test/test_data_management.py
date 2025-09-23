import sys, os

sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from jax import random, tree_util
import jax.numpy as jnp
from src.neural_network import CNNDecoder, mCNNDecoder
from src.data_management import save_params, load_params, load_model
import tempfile


def _params_equal(p1, p2):
    return tree_util.tree_all(
        tree_util.tree_map(lambda a, b: jnp.allclose(a, b), p1, p2)
    )

def test_save_and_load_params():
	# Arrange
	key = random.key(3)
	input_shape=(2, 4, 4)
	conv_layers=[(2, 2, 1, 0)]
	fc_layers=[8, 3]
	# Act
	model = CNNDecoder(
		input_shape,
		conv_layers,
		fc_layers
	)
	params = model.init(key)
	# Assert
	with tempfile.TemporaryDirectory() as tmpdir:
		file_name = os.path.join(tmpdir, 'params.json')
		save_params(file_name, params)
		assert os.path.exists(file_name)
		loaded_params = load_params(file_name)
		assert _params_equal(loaded_params, params)

def test_load_model():
    # Arrange
    model_name = "test"
    # Act
    model, model_params, prefered_error_probabilities = load_model(model_name)
    # Assert
    assert isinstance(model, mCNNDecoder)
    assert prefered_error_probabilities.sum() == 1.0
    assert tree_util.tree_all(
        tree_util.tree_map(lambda a: isinstance(a, jnp.ndarray), model_params)
    )


if __name__ == "__main__":
    test_save_and_load_params()
    test_load_model()
    print("All tests passed!")
