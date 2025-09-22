import sys, os
sys.path.append(os.path.abspath(
    os.getcwd()+"/experiments/src")
)
# Training imports
import jax.numpy as jnp
from jax import random, lax, nn, jit, vmap, value_and_grad
import optax
# Timing and data saving
from time import perf_counter
import json
# Quantum error correction imports
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.generic import BiasedDepolarizingErrorModel
# Local imports
from src.neural_network import CNNDual, save_params
from src.data_gen import sample_error_batch, sample_deformation_batch, transform_code_stabilizers, syndrome_to_image_mapper, deformation_to_image_mapper, relevancy_tensor


# NOTE: Variables in ALL_CAPS are global constants and should not be changed outside of the initial setup section


# Command line arguments
if len(sys.argv) < 6 or len(sys.argv) > 7:
    raise ValueError("Please provide the following command line arguments: <save_decoder_as> <deformation_name> <code_distance> <training_config> <training_batches> [random_seed]")
NAME = sys.argv[1]
deformation_name = sys.argv[2]
CODE_DISTANCE = int(sys.argv[3])
training_config = sys.argv[4]
TRAINING_BATCHES = int(sys.argv[5])
SEED = int(sys.argv[6]) if len(sys.argv) == 7 else 0

settings = {
    "<file_name>": os.path.relpath(sys.argv[0]),
    "<save_decoder_as>": sys.argv[1],
    "<deformation_name>": sys.argv[2],
    "<code_distance>": sys.argv[3],
    "<training_config>": sys.argv[4],
    "<training_batches>": sys.argv[5],
}
if len(sys.argv) == 7:
    settings["<random_seed>"] = sys.argv[6]

# Create results directories if they don't exist
save_dir = f"results/{NAME}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
else:
    input(f"Warning: A saved NN under the name \"{NAME}\" already exists. Press Enter to continue and override it, or Ctrl+C to abort.")

# Parameters
with open(f"experiments/training_configs/{training_config}.json", "r") as f:
    config: dict = json.load(f)
BATCH_SIZE = config["BATCH_SIZE"]
INIT_LEARNING_RATE = config["INIT_LEARNING_RATE"]
LEARNING_RATE = config["LEARNING_RATE"]
WARMUP_STEPS = config["WARMUP_STEPS"]
TRANSITION_STEP = config["TRANSITION_STEP"]
DECAY_RATE = config["DECAY_RATE"]
ERROR_PROBABILITY = config["ERROR_PROBABILITY"]
ERROR_BIAS = config["ERROR_BIAS"]

# Set up NN architecture
NUM_FILTERS = CODE_DISTANCE**2-1
CONV_LAYERS_INPUT_1 = [(NUM_FILTERS,2,1,0)]
CONV_LAYERS_INPUT_2 = [(NUM_FILTERS,1,1,0)]
CONV_LAYERS_STAGE_2 = [(NUM_FILTERS,2,1,0)]
FC_LAYERS = [50, 2]
nn_decoder = CNNDual(
    input_shape_1 = (1, CODE_DISTANCE+1, CODE_DISTANCE+1),
    input_shape_2 = (6, CODE_DISTANCE, CODE_DISTANCE),
    conv_layers_input_1 = CONV_LAYERS_INPUT_1,
    conv_layers_input_2 = CONV_LAYERS_INPUT_2,
    conv_layers_stage_2 = CONV_LAYERS_STAGE_2,
    fc_layers = FC_LAYERS
)

# Set up learning rate schedule
learning_rate = optax.warmup_exponential_decay_schedule(
    init_value=INIT_LEARNING_RATE,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    transition_steps=TRANSITION_STEP,
    decay_rate=DECAY_RATE
)

# Setup code, error model, and deformation
code = RotatedPlanarCode(CODE_DISTANCE, CODE_DISTANCE)
ERROR_MODEL = BiasedDepolarizingErrorModel(ERROR_BIAS, axis="Z")
match deformation_name:
    case "Generalized":
        DEFORMATION = "Generalized"
    case "Best":
        DEFORMATION = "Best"
    case "CSS":
        DEFORMATION = jnp.zeros(CODE_DISTANCE**2, dtype=jnp.int32)
    case "XZZX":
        DEFORMATION = jnp.zeros(CODE_DISTANCE**2, dtype=jnp.int32).at[::2].set(3)
    case "XY":
        DEFORMATION = jnp.zeros(CODE_DISTANCE**2, dtype=jnp.int32).at[:].set(2)
    case "C1":
        DEFORMATION = jnp.zeros((CODE_DISTANCE, CODE_DISTANCE), dtype=jnp.int32).at[1::2, ::2].set(3).flatten().at[::2].set(2)
    case _:
        if all(char in "012345" for char in deformation_name) and len(deformation_name) == CODE_DISTANCE**2:
            DEFORMATION = jnp.array([int(char) for char in deformation_name], dtype=jnp.int32)
        else:
            raise ValueError(f"Unknown deformation_name: {deformation_name}")
if isinstance(DEFORMATION, str):
    print(f"Mode \"{DEFORMATION}\" selected.")
else:
    print(f"Deformation:\n{DEFORMATION.reshape((CODE_DISTANCE,CODE_DISTANCE))}")

# Image mappers
SYNDROME_TO_IMAGE_MAP = syndrome_to_image_mapper(code)
DEFORMATION_TO_IMAGE_MAP = deformation_to_image_mapper(code)

# Setup the training routines
def train(
    code: RotatedPlanarCode,
    optimizer: optax.GradientTransformation,
    model: CNNDual,
    pretrained_model_params: dict | None = None,
):
    @jit
    def loss_fn(
        model_params,
        deformations: jnp.ndarray,
        errors: jnp.ndarray,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        y: jnp.ndarray,
    ):
        """
        Loss function for the model. The loss is the binary cross entropy (BCE) between the
        model output and the logicals. The BCE is weighted by the probability of the deformation.
        """
        # Calculate the BCE
        idv_loss = optax.sigmoid_binary_cross_entropy(
            logits=model.apply_batch(model_params, x1, x2),
            labels=y
        ).mean(axis=1)
        # If DEFORMATION is set, we don't use weights
        # and just return the mean BCE
        if isinstance(DEFORMATION, jnp.ndarray) or DEFORMATION == "Generalized":
            return jnp.mean(idv_loss)
        # Calculate the weights for the BCE
        probs = nn.softmax(model_params["deformation_dist"], axis=0).reshape(6, -1).T[None, :, :]
        n = errors.shape[1] // 2
        err_idx = (errors[:,:n] + 2*errors[:,n:])
        # The sum is over the deformation probabilities on the same data qubit
        # The prod is over the data qubits for the same batch
        weights = jnp.prod(jnp.sum(probs*relevancy_tensor[err_idx, deformations], axis=2), axis=1)
        # Weights are applied to the BCE
        # 1 is added to all weights in the scenario that they all sum to less than 0.01 as to avoid devision by 0
        return jnp.average(idv_loss, weights=weights + (weights.sum() < .01).astype(jnp.float32))

    @jit
    def update(
        model_params, 
        deformations: jnp.ndarray,
        errors: jnp.ndarray,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        y: jnp.ndarray,
        opt_state,
    ):
        mse_loss_batch_val_grad = value_and_grad(loss_fn, argnums=0)
        mse_loss_batch_val_grad = jit(mse_loss_batch_val_grad)
        loss, grads = mse_loss_batch_val_grad(model_params, deformations, errors, x1, x2, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)
        return loss, model_params, opt_state

    @jit
    def get_data(
        data_key,
        deformations: jnp.ndarray,
    ):
        subkey, data_key = random.split(data_key)
        stabilizers, logicals = vmap(transform_code_stabilizers, in_axes=(None, 0))(code, deformations)
        errors = sample_error_batch(
            subkey,
            BATCH_SIZE,
            code,
            ERROR_MODEL,
            ERROR_PROBABILITY,
        )
        syndromes = vmap(lambda s,e: (s @ e) % 2, in_axes=(0, 0))(stabilizers, errors)
        logical_errors = vmap(lambda l,e: (l @ e) % 2, in_axes=(0, 0))(logicals, errors)
        syndrome_images = vmap(SYNDROME_TO_IMAGE_MAP)(syndromes)
        return syndrome_images, logical_errors, errors, data_key

    def _fori_body(
        i: int,
        val: tuple
    ):
        (
            model_params,
            opt_state,
            data_key,
            deformation_key,
            losses,
        ) = val

        # The if statement will be jit compiled away on execution
        # deformations shape=(BATCH_SIZE, NUM_DATA_QUBITS)
        if isinstance(DEFORMATION, jnp.ndarray):
            # If DEFORMATION is set, we only train on that deformation
            deformations = jnp.tile(DEFORMATION, reps=(BATCH_SIZE, 1))
        elif DEFORMATION == "Best":
            # Sample a deformation for each batch
            probs = nn.softmax(model_params["deformation_dist"], axis=0)
            subkey, deformation_key = random.split(deformation_key)
            deformations = sample_deformation_batch(subkey, BATCH_SIZE, probs)
        elif DEFORMATION == "Generalized":
            # Sample a deformation for each batch with uniform distribution
            probs = jnp.ones_like(model_params["deformation_dist"]) / 6
            subkey, deformation_key = random.split(deformation_key)
            deformations = sample_deformation_batch(subkey, BATCH_SIZE, probs)
        # deformation_images shape=(BATCH_SIZE, 6, CODE_DISTANCE, CODE_DISTANCE)
        deformation_images = vmap(DEFORMATION_TO_IMAGE_MAP)(deformations)

        # syndrome_images shape=(BATCH_SIZE, 1, CODE_DISTANCE+1, CODE_DISTANCE+1)
        # logicals shape=(BATCH_SIZE, 2)
        syndrome_images, logicals, errors, data_key = get_data(
            data_key,
            deformations,
        )
        loss, model_params, opt_state = update(
            model_params,
            deformations,
            errors,
            syndrome_images,
            deformation_images,
            logicals,
            opt_state,
        )
        losses = losses.at[i].set(loss)
        # Recentering the deformation distribution to avoid numerical issues (the softmax function is invariant to additive constants)
        model_params["deformation_dist"] = model_params["deformation_dist"] - model_params["deformation_dist"].mean(axis=0)

        return (
            model_params,
            opt_state,
            data_key,
            deformation_key,
            losses,
        )
    
    init_key, data_key, deformation_key = random.split(
        key = random.key(SEED),
        num=3
    )

    if pretrained_model_params is None:
        # Initialize the model parameters
        model_params = model.init(init_key)
    else:
        # Copy the model parameters
        model_params = pretrained_model_params.copy()
    # The deformation distribution is initialized to be uniform over all deformations
    if isinstance(DEFORMATION, jnp.ndarray):
        # If DEFORMATION is set, we only train on that deformation
        # As model_params["deformation_dist"] will be turned into a probability distribution using the softmax function
        # we make sure that the chosen deformation will end up with 100% probability.
        model_params["deformation_dist"] = 1000*(DEFORMATION_TO_IMAGE_MAP(DEFORMATION) - .5)
    else:
        # If DEFORMATION is not set, we train on all deformations
        # and the deformation distribution is initialized to be uniform over all deformations
        model_params["deformation_dist"] = jnp.zeros(
            shape=(6, CODE_DISTANCE, CODE_DISTANCE),
            dtype=jnp.float32
        )
    optimizer_state = optimizer.init(model_params)

    losses = jnp.zeros(shape=TRAINING_BATCHES, dtype=jnp.float32)

    val_init = (
        model_params,
        optimizer_state,
        data_key,
        deformation_key,
        losses,
    )

    vals = lax.fori_loop(0, TRAINING_BATCHES, _fori_body, val_init)

    return vals

# Perform the training
print("Starting training...")
start_time = perf_counter()
(
    model_params,
    opt_state,
    data_key,
    deformation_key,
    losses,
) = train(
    code=code,
    optimizer=optax.adam(learning_rate),
    model=nn_decoder,
    pretrained_model_params=None,
)
losses.block_until_ready()  # Wait for training to finish
end_time = perf_counter()
training_time = end_time - start_time
print(f"Training finished in {int(training_time/60/60):d}h {int(training_time/60%60):02d}m {int(training_time%60):02d}s")

# Save the data
jnp.save(
    f"{save_dir}/loss.npy",
    losses,
)
save_params(
    f"{save_dir}/model_params.json",
    model_params,
)
with open(f"{save_dir}/settings.json", "w") as f:
    json.dump(settings, f, indent=4)
print("Data saved under:", save_dir)