import sys, os

sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments")
)

from src.neural_network_modules import MLPBlock, CNNBlock, Embedder, PositionalEncodingND, TransformerEncoder, DecoderPredBlock
from flax import linen as nn
import jax.numpy as jnp


def print_params_structure(params, indent=0):
    for key in params.keys():
        print("  " * indent + str(key), end="")
        if isinstance(params[key], dict):
            print()
            print_params_structure(params[key], indent + 1)
        else:
            print(f":\t shape {params[key].shape}")


class CNNDecoder(nn.Module):
    layer_sizes: list[int] # List of layer sizes
    conv_features: list[int]  # Number of output channels for each convolutional layer
    kernel_sizes: list[int]   # Kernel sizes for each convolutional layer
    layer_padding: str = 'Valid'  # Padding type for convolutional layers
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.relu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.cnn_block = CNNBlock(
            conv_features=self.conv_features,
            kernel_sizes=self.kernel_sizes,
            layer_padding=self.layer_padding,
            input_dropout_rate=self.input_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.mlp_block = MLPBlock(
            layer_sizes=self.layer_sizes,
            input_dropout_rate=self.internal_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
    
    def __call__(self, x):
        x = self.cnn_block(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output for MLP input
        x = self.mlp_block(x)
        x = nn.softmax(x)
        return x


class mCNNDecoder(nn.Module):
    layer_sizes: list[int] # List of layer sizes
    conv_features: dict[str, list[int]]  # Number of output channels for each convolutional layer
    kernel_sizes: dict[str, list[int]]  # Kernel sizes for each convolutional layer
    layer_padding: dict[str, str] = 'Valid'  # Padding type for convolutional layers
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.relu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.cnn_syndrome_block = CNNBlock(
            conv_features=self.conv_features['synd'],
            kernel_sizes=self.kernel_sizes['synd'],
            layer_padding=self.layer_padding['synd'],
            input_dropout_rate=self.input_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.cnn_deformation_block = CNNBlock(
            conv_features=self.conv_features['defo'],
            kernel_sizes=self.kernel_sizes['defo'],
            layer_padding=self.layer_padding['defo'],
            input_dropout_rate=self.input_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.cnn_internal_block = CNNBlock(
            conv_features=self.conv_features['internal'],
            kernel_sizes=self.kernel_sizes['internal'],
            layer_padding=self.layer_padding['internal'],
            input_dropout_rate=self.internal_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.mlp_block = MLPBlock(
            layer_sizes=self.layer_sizes,
            input_dropout_rate=self.internal_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
    
    def __call__(self, x_s, x_d):
        x_s = self.cnn_syndrome_block(x_s)
        x_d = self.cnn_deformation_block(x_d)
        x = x_s * x_d # Combine the two inputs element-wise
        x = self.cnn_internal_block(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output for MLP input
        x = self.mlp_block(x)
        x = nn.softmax(x)
        return x


class TransformerDecoder(nn.Module):
    site_locations: jnp.ndarray  # Array of shape (num_sites, dim) containing (x, y, ...) coordinates of each site
    output_features: int  # Output feature size
    vocab_size: int  # Size of the vocabulary
    num_layers: int  # Number of encoder layers
    heads: int  # Number of attention heads
    d_model: int  # Dimension of the model
    mlp_dim: int  # Internal dimension of the feedforward network
    pooling: str = 'avg'  # Pooling type: 'avg' or 'max'
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.gelu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.embedder = Embedder(
            vocab_size=self.vocab_size,
            features=self.d_model
        )
        self.positional_encoding = PositionalEncodingND(
            d_model=self.d_model, 
            site_locations=self.site_locations
        )
        self.transformer_first_round = TransformerEncoder(
            num_layers=self.num_layers,
            heads=self.heads,
            d_model=self.d_model,
            mlp_dim=self.mlp_dim,
            input_dropout_rate=self.input_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.transformer_internal_round = TransformerEncoder(
            num_layers=self.num_layers,
            heads=self.heads,
            d_model=self.d_model,
            mlp_dim=self.mlp_dim,
            input_dropout_rate=self.input_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
        self.decoder = DecoderPredBlock(
            features=self.output_features,
            pooling=self.pooling
        )
    
    def __call__(self, x_init, x_internal):
        """
        x_init: (batch_size, num_sites)
        x_internal: (batch_size, num_rounds, num_sites)
        state: (batch_size, num_sites, d_model)
        x: (batch_size, output_features)
        """
        # // First round //
        x_first_round = self.embedder(x_init)
        x_first_round = self.positional_encoding(x_first_round)
        state = self.transformer_first_round(x_first_round)
        # // Internal rounds //
        for r in range(x_internal.shape[1]):
            x_internal_round = self.embedder(x_internal[:, r, :])
            x_internal_round = self.positional_encoding(x_internal_round)
            state = (state + x_internal_round) * 0.7
            state = self.transformer_internal_round(state)
        # // Final prediction //
        x = self.decoder(state)
        return x
    
    def apply_first_round(self, x_init):
        """
        x_init: (batch_size, num_sites)
        state: (batch_size, num_sites, d_model)
        """
        x_first_round = self.embedder(x_init)
        x_first_round = self.positional_encoding(x_first_round)
        state = self.transformer_first_round(x_first_round)
        return state

    def apply_internal_round(self, state, x_internal_round):
        """
        state: (batch_size, num_sites, d_model)
        x_internal_round: (batch_size, num_sites)
        """
        x_internal_round = self.embedder(x_internal_round)
        x_internal_round = self.positional_encoding(x_internal_round)
        state = (state + x_internal_round) * 0.7
        state = self.transformer_internal_round(state)
        return state
    
    def apply_final_prediction(self, state):
        """
        state: (batch_size, num_sites, d_model)
        """
        x = self.decoder(state)
        return x

if __name__ == "__main__":
    import jax.numpy as jnp
    from jax import random
    from qecsim.models.rotatedplanar import RotatedPlanarCode

    key = random.key(0)

    code_distance = 3
    code = RotatedPlanarCode(code_distance, code_distance)
    plaquette_coords = code._plaquette_indices
    data_qubit_coords = [(x-.5, y-.5) for y in range(code_distance) for x in range(code_distance)]

    batch_size = 5
    syndromes = random.randint(key, (batch_size, code_distance**2-1), 0, 2)  # Example syndrome input {0, 1}
    deformation = random.randint(key, (batch_size, code_distance**2), 0, 6) + 2  # Example deformation input {2, 3, 4, 5, 6, 7}
    x = jnp.append(syndromes, deformation, axis=1)

    model = TransformerDecoder(
        site_locations=jnp.array(plaquette_coords + data_qubit_coords),
        output_features=4,
        vocab_size=8,
        num_layers=2,
        heads=4,
        d_model=32,
        mlp_dim=128,
        training=True
    )

    params = model.init(key, x)

    print_params_structure(params)

    print("\nInput:\n", x)
    y = model.apply(params, x, rngs=key)
    print("\nOutput:\n", y)