import jax.numpy as jnp
from flax import linen as nn
from jax import device_put


class MLPBlock(nn.Module):
    layer_sizes: list[int] # List of layer sizes
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.relu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.fc_layer = [nn.Dense(size) for size in self.layer_sizes]
        self.input_dropout = nn.Dropout(rate=self.input_dropout_rate)
        self.internal_dropout = nn.Dropout(rate=self.internal_dropout_rate)

    def __call__(self, x):
        x = self.input_dropout(x, deterministic=not self.training)
        for layer in self.fc_layer[:-1]:
            x = layer(x)
            x = self.activation_fun(x)
            x = self.internal_dropout(x, deterministic=not self.training)
        x = self.fc_layer[-1](x)
        return x


class GatedMLPBlock(nn.Module):
    layer_sizes: list[int] # List of layer sizes
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.relu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        assert all(size % 2 == 0 for size in self.layer_sizes[:-1]), "All hidden layer sizes must be even for gated activations."
        self.fc_layer = [nn.Dense(size) for size in self.layer_sizes]
        self.input_dropout = nn.Dropout(rate=self.input_dropout_rate)
        self.internal_dropout = nn.Dropout(rate=self.internal_dropout_rate)

    def __call__(self, x):
        x = self.input_dropout(x, deterministic=not self.training)
        for layer in self.fc_layer[:-1]:
            x = layer(x)
            a, b = jnp.split(x, 2, axis=-1)  # Split the output into two halves
            x = self.activation_fun(a) * b  # Gated activation
            x = self.internal_dropout(x, deterministic=not self.training)
        x = self.fc_layer[-1](x)
        return x


class CNNBlock(nn.Module):
    conv_features: list[int]  # Number of output channels for each convolutional layer
    kernel_sizes: list[int]   # Kernel sizes for each convolutional layer
    layer_padding: str = 'Valid'  # Padding type for convolutional layers
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.relu  # Activation function
    training: bool = True  # Whether to use deterministic (no dropout) mode

    def setup(self):
        self.conv_layers = [nn.Conv(features=feat, kernel_size=(k, k), strides=(1, 1), padding=self.layer_padding)
                            for feat, k in zip(self.conv_features, self.kernel_sizes)]
        self.input_dropout = nn.Dropout(rate=self.input_dropout_rate)
        self.internal_dropout = nn.Dropout(rate=self.internal_dropout_rate)

    def __call__(self, x):
        x = self.input_dropout(x, deterministic=not self.training)
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation_fun(x)
            x = self.internal_dropout(x, deterministic=not self.training)
        return x


class Embedder(nn.Module):
    vocab_size: int  # Size of the vocabulary
    features: int        # Dimension of the embedding vectors

    def setup(self):
        self.embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.features,
            embedding_init=nn.initializers.normal(stddev=1.0)
        )

    def __call__(self, x):
        original_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # Flatten the last three dimensions
        x = self.embedder(x)
        x = x.reshape((*original_shape, self.features))  # Restore original shape
        return x


class PositionalEncoding3D(nn.Module):
    d_model: int  # Dimension of the model (must be divisible by 6)
    max_len: int = 100  # Maximum length of the sequence

    def setup(self):
        assert self.d_model % 6 == 0, "Channel dimension must be divisible by 6"

        B = self.max_len # Default value is 10000.0
        div_term = jnp.exp(jnp.arange(0, self.d_model // 6) * -(jnp.log(B) * 6 / self.d_model))

        # Create grid of coordinates
        x_pos = jnp.arange(0, self.max_len, dtype=jnp.float32)
        y_pos = jnp.arange(0, self.max_len, dtype=jnp.float32)
        z_pos = jnp.arange(0, self.max_len, dtype=jnp.float32)
        xx, yy, zz = jnp.meshgrid(x_pos, y_pos, z_pos, indexing='ij')  # shape (w, h, d)

        # Compute positional encodings for each axis
        pe = jnp.zeros((self.max_len, self.max_len, self.max_len, self.d_model))
        third = self.d_model // 3
        # X axis
        for i in range(third // 2):
            pe = pe.at[..., 2*i   + 0*third].set(jnp.sin(xx * div_term[i]))
            pe = pe.at[..., 2*i+1 + 0*third].set(jnp.cos(xx * div_term[i]))
        # Y axis
        for j in range(third // 2):
            pe = pe.at[..., 2*j   + 1*third].set(jnp.sin(yy * div_term[j]))
            pe = pe.at[..., 2*j+1 + 1*third].set(jnp.cos(yy * div_term[j]))
        # Z axis
        for k in range(third // 2):
            pe = pe.at[..., 2*k   + 2*third].set(jnp.sin(zz * div_term[k]))
            pe = pe.at[..., 2*k+1 + 2*third].set(jnp.cos(zz * div_term[k]))
        
        pe = pe[None, :, :, :, :]  # Add batch dimension
        self.pe = device_put(pe)  # Move to device

    def __call__(self, x):
        b, w, h, d, c = x.shape
        x = x + self.pe[:, :w, :h, :d, :]
        return x.reshape(b, w*h*d, c)


# class PositionalEncoding2D(nn.Module):
#     d_model: int # Dimension of the model (should be a multiple of 4)
#     site_locations: jnp.ndarray  # Array of shape (num_sites, dim) containing (x, y) coordinates of each site

#     def setup(self):
#         assert self.d_model % 4 == 0, "d_model must be a multiple of 4"

#         pe = jnp.zeros((self.site_locations.shape[0], self.d_model))

#         x = self.site_locations[:, 0]
#         y = self.site_locations[:, 1]

#         B = self.site_locations.max() - self.site_locations.min() # Default is 10000.0
#         div_term = jnp.exp(jnp.arange(0, self.d_model // 4) * -(jnp.log(B) / self.d_model))

#         split = self.d_model // 2
#         # Use the first half of the dimensions for x,
#         pe = pe.at[:, 0:split:2].set(jnp.sin(x[:, None] * div_term)) # Even indices
#         pe = pe.at[:, 1:split:2].set(jnp.cos(x[:, None] * div_term)) # Odd indices
#         # and the second half for y
#         pe = pe.at[:, split::2].set(jnp.sin(y[:, None] * div_term)) # Even indices
#         pe = pe.at[:, split+1::2].set(jnp.cos(y[:, None] * div_term)) # Odd indices

#         pe = pe[None, :, :]  # Add batch dimension
#         self.pe = device_put(pe)  # Move to device

#     def __call__(self, x):
#         x = x + self.pe
#         return x
class PositionalEncodingND(nn.Module):
    d_model: int # Dimension of the model (should be a multiple of 4)
    site_locations: jnp.ndarray  # Array of shape (num_sites, dim) containing (x, y, ...) coordinates of each site

    def setup(self):
        assert self.d_model % (2*self.site_locations.shape[1]) == 0, f"d_model must be a multiple of {2*self.site_locations.shape[1]}"

        pe = jnp.zeros((self.site_locations.shape[0], self.d_model))

        B = self.site_locations.max() - self.site_locations.min() # Default is 10000.0
        div_term = jnp.exp(jnp.arange(0, self.d_model // 4) * -(jnp.log(B) / self.d_model))

        splits = self.d_model // self.site_locations.shape[1]
        for i in range(self.site_locations.shape[1]):
            start, end = i*splits, (i+1)*splits
            axis = self.site_locations[:, i]
            # Use the i-th split of the dimensions for the i-th axis
            pe = pe.at[:, start:end:2].set(jnp.sin(axis[:, None] * div_term)) # Even indices
            pe = pe.at[:, start+1:end:2].set(jnp.cos(axis[:, None] * div_term)) # Odd indices

        pe = pe[None, :, :]  # Add batch dimension
        self.pe = device_put(pe)  # Move to device

    def __call__(self, x):
        x = x + self.pe
        return x


class EncoderBlock(nn.Module):
    heads: int  # Number of attention heads
    d_model: int  # Dimension of the model
    mlp_dim: int  # Internal dimension of the feedforward network
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.gelu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.attention = nn.MultiHeadAttention(
            num_heads=self.heads,
            qkv_features=self.d_model,
            dropout_rate=self.internal_dropout_rate,
        )

        self.norm_attention = nn.LayerNorm()
        self.norm_mlp = nn.LayerNorm()

        self.gated_mlp = GatedMLPBlock(
            layer_sizes=[self.mlp_dim, self.d_model],
            input_dropout_rate=self.internal_dropout_rate,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        )
    
    def __call__(self, x):
        # Multi-head self-attention
        attn_output = self.attention(self.norm_attention(x), deterministic=not self.training)
        x = x + attn_output

        # Feedforward network
        mlp_output = self.gated_mlp(self.norm_mlp(x))
        x = x + mlp_output

        return x


class TransformerEncoder(nn.Module):
    num_layers: int  # Number of encoder layers
    heads: int  # Number of attention heads
    d_model: int  # Dimension of the model
    mlp_dim: int  # Internal dimension of the feedforward network
    input_dropout_rate: float = 0.08  # Dropout rate applied to the input
    internal_dropout_rate: float = 0.5  # Dropout rate applied after each layer
    activation_fun: callable = nn.gelu  # Activation function
    training: bool = True  # Set to True during training, False during evaluation

    def setup(self):
        self.encoder_layers = [EncoderBlock(
            heads=self.heads,
            d_model=self.d_model,
            mlp_dim=self.mlp_dim,
            internal_dropout_rate=self.internal_dropout_rate,
            activation_fun=self.activation_fun,
            training=self.training
        ) for _ in range(self.num_layers)]
        self.input_dropout = nn.Dropout(rate=self.input_dropout_rate)
    
    def __call__(self, x):
        x = self.input_dropout(x, deterministic=not self.training)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class DecoderPredBlock(nn.Module):
    features: int  # Number of output features
    pooling: str = 'avg'  # Pooling type: 'avg' or 'max'

    def setup(self):
        assert self.pooling in ['avg', 'max'], "Pooling must be either 'avg' or 'max'"
        if self.pooling == 'avg':
            self.pool = jnp.mean
        elif self.pooling == 'max':
            self.pool = jnp.max
        else:
            raise ValueError(f"Pooling type '{self.pooling}' is not supported. Use 'avg' or 'max'.")
        self.fc = nn.Dense(self.features)
    
    def __call__(self, x):
        # Assume x has shape (batch_size, seq_length, d_model)
        x = self.pool(x, axis=1)  # Pooling
        x = self.fc(x)  # Fully connected layer
        x = nn.softmax(x)  # Interpret as probabilities
        return x


if __name__ == "__main__":
    from jax import random

    L = 3
    batch_size = 4
    d_model = 64

    key_synd, key_defo, key = random.split(random.key(0), 3)
    syndromes = random.randint(key_synd, (batch_size, L**2-1), 0, 2)
    deformation = random.randint(key_defo, (batch_size, 2*L), 0, 6)

    print("Syndrome:\n", syndromes)
    print("Deformation:\n", deformation)
    
