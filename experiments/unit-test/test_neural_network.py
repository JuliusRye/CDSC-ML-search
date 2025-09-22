import sys, os

import jax
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

import tempfile
import jax.numpy as jnp
from jax import random
from src.neural_network import MLP, CNN, mCNNDecoder, CNNDecoder, save_params, load_params

def test_mlp_forward():
	# Arrange
	model = MLP(layer_sizes=[4, 6, 4, 5])
	x_single = jnp.array([2.2, 5.1, 1.6, 6.2])
	x_batch = jnp.array([[2.2, 5.1, 1.6, 6.2], [2.2, 5.5, 1.6, 6.2]])
	key = random.key(0)
	# Act
	params = model.init(key)
	out_single = model.apply_single(params, x_single)
	out_batch = model.apply_batch(params, x_batch)
	# Assert
	assert out_single.shape == (5,)
	assert out_batch.shape == (2, 5)

def test_cnn_forward():
	# Arrange
	key = random.key(1)
	input_shape=(3, 5, 5)
	conv_layers=[(1, 3, 2, 2)]
	batch_size = 2
	channels = 3
	# Act
	model = CNN(input_shape=input_shape, conv_layers=conv_layers)
	params = model.init(key)
	img = random.uniform(key, shape=(batch_size, channels, 5, 5))
	out = model.apply_batch(params, img)
	# Assert
	assert out.shape == (batch_size, 1, 4, 4)

def test_cnndual_forward():
	# Arrange
	key = random.key(2)
	input_shape_1=(1, 4, 4)
	input_shape_2=(6, 3, 3)
	conv_layers_input_1=[(5, 2, 1, 0)]
	conv_layers_input_2=[(5, 1, 1, 0)]
	conv_layers_stage_2=[(10, 2, 1, 0)]
	fc_layers=[25, 5]
	batch_size = 2
	# Act
	model = mCNNDecoder(
		input_shape_1,
		input_shape_2,
		conv_layers_input_1,
		conv_layers_input_2,
		conv_layers_stage_2,
		fc_layers,
	)
	params = model.init(key)
	img_prim = random.uniform(key, shape=(batch_size, 1, 4, 4))
	img_seco = random.uniform(key, shape=(batch_size, 6, 3, 3))
	out_batch = model.apply_batch(params, img_prim, img_seco)
	# Assert
	assert out_batch.shape == (batch_size, 5)

def test_cnndecoder_forward():
	# Arrange
	key = random.key(3)
	input_shape=(2, 4, 4)
	conv_layers=[(2, 2, 1, 0)]
	fc_layers=[8, 3]
	batch_size = 2
	# Act
	model = CNNDecoder(
		input_shape,
		conv_layers,
		fc_layers
	)
	params = model.init(key)
	img = random.uniform(key, shape=(batch_size, 2, 4, 4))
	out_batch = model.apply_batch(params, img)
	# Assert
	assert out_batch.shape == (batch_size, 3)

def _params_equal(p1, p2):
    return jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), p1, p2)
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

if __name__ == "__main__":
    test_mlp_forward()
    test_cnn_forward()
    test_cnndual_forward()
    test_cnndecoder_forward()
    test_save_and_load_params()
    print("All tests passed!")
	