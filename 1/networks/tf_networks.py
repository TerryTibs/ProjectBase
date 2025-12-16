# --- START OF FILE project/networks/tf_networks.py ---

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp # For policy gradient distributions

# Import custom layers specific to TF using the new path
# Ensure this path is correct relative to where tf_networks.py is located
try:
    # Assuming tf_networks.py is in project/networks/
    from ..layers.custom_layers_tf import NoisyLinear, MeanReducer
except ImportError:
    # Fallback if running script directly from networks dir (less ideal)
    from layers.custom_layers_tf import NoisyLinear, MeanReducer


# --- Custom Layer for Stacking ---
class StackLayer(layers.Layer):
    """ A custom Keras layer to wrap tf.stack. """
    def __init__(self, axis=1, **kwargs):
        super(StackLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def get_config(self):
        config = super(StackLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
# --- End Custom Layer ---


# --- Helper Function for Dense Blocks ---
def _create_dense_block(input_tensor, hidden_units, activation, LayerType, layer_kwargs, name_prefix):
    """Helper to create a block of dense or noisy layers with activation handling."""
    x = input_tensor
    for i, units in enumerate(hidden_units):
        layer_name = f'{name_prefix}_hidden_{i+1}'
        # Use issubclass for robust type checking
        if issubclass(LayerType, NoisyLinear):
            x = LayerType(units, **layer_kwargs, name=layer_name)(x)
            # Apply activation *after* NoisyLinear layer
            if activation == 'relu':
                x = layers.ReLU(name=f'{layer_name}_relu')(x)
            elif activation == 'tanh':
                 x = layers.Activation('tanh', name=f'{layer_name}_tanh')(x)
            # Add elif for other activations if needed
            elif activation is not None and activation != 'linear':
                 # Generic activation layer if specified and not linear
                 x = layers.Activation(activation, name=f'{layer_name}_{activation}')(x)
            # No activation layer needed if activation is None or 'linear'
        else: # Assume standard Dense layer
             # Pass activation directly to the Dense layer constructor
            x = LayerType(units, activation=activation, name=layer_name)(x)
    return x


# --- CNN Feature Extractor (Handles 2D Grid or 3D Pixels) ---
def build_cnn_feature_extractor(input_shape, activation='relu', name="CNN_FeatureExtractor", input_dtype=tf.uint8):
    """
    Builds the convolutional part of a CNN model.
    Handles 2D grid (float32) or 3D pixel (uint8) inputs automatically.
    """
    is_grid_input = False
    if len(input_shape) == 2: # 2D Grid state (Rows, Cols)
        actual_input_shape = (*input_shape, 1) # Add channel dimension
        is_grid_input = True
        print(f"Info: Received 2D input shape {input_shape}, using {actual_input_shape} for CNN.")
    elif len(input_shape) == 3: # 3D Pixel state (H, W, C)
        actual_input_shape = input_shape
    else:
        raise ValueError(f"{name} expects (H, W, C) or (Rows, Cols), got {input_shape}")

    # Ensure correct dtype if grid input
    if is_grid_input and input_dtype != tf.float32:
        print(f"Warning: Grid input detected, but input_dtype was {input_dtype}. Forcing tf.float32.")
        input_dtype = tf.float32

    # Define Input layer with correct shape and dtype
    input_layer = layers.Input(shape=actual_input_shape, dtype=input_dtype, name=f'{name}_input')

    # Conditional Normalization based on input dtype
    if input_dtype == tf.uint8: # Apply normalization only for pixel inputs
        norm_layer = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0, name=f'{name}_normalize')(input_layer)
        conv_input = norm_layer
        print(f"Info: {name} applying pixel normalization (input dtype: {input_dtype}).")
    else: # Assume float input (grid state), skip normalization
        conv_input = input_layer
        print(f"Info: {name} skipping pixel normalization (input dtype: {input_dtype}).")

    # Standard CNN layers (Nature DQN style)
    conv1 = layers.Conv2D(32, kernel_size=8, strides=4, activation=activation, name=f'{name}_conv1', padding='same')(conv_input)
    conv2 = layers.Conv2D(64, kernel_size=4, strides=2, activation=activation, name=f'{name}_conv2', padding='same')(conv1)
    conv3 = layers.Conv2D(64, kernel_size=3, strides=1, activation=activation, name=f'{name}_conv3', padding='same')(conv2)
    flatten_layer = layers.Flatten(name=f'{name}_flatten')(conv3)

    # Create and return the feature extractor model
    model = models.Model(inputs=input_layer, outputs=flatten_layer, name=name)
    print(f"\n--- Built Network: {name} (Input: {actual_input_shape}, dtype: {input_dtype}) ---")
    model.summary(line_length=100)
    print("-" * (len(name) + 40))
    return model


# --- DQN Head Builder (for CNNs) ---
def build_cnn_dqn_head(input_tensor, num_actions, noisy=True, noisy_std_init=0.5,
                       fc_units=512, activation='relu', name="CNN_Head"):
    """Builds the fully connected head for a CNN DQN (takes flattened CNN features)."""
    LayerType = NoisyLinear if noisy else layers.Dense
    layer_kwargs = {'std_init': noisy_std_init} if noisy else {}

    # Hidden FC layer using helper
    fc_output = _create_dense_block(input_tensor, [fc_units], activation, LayerType, layer_kwargs, f"{name}_fc")

    # Final output layer for Q-values (linear activation)
    output_layer = NoisyLinear if noisy else layers.Dense
    output_kwargs = {'std_init': noisy_std_init} if noisy else {}
    # Ensure linear activation for the final Q-value layer if using Dense
    if not noisy:
        output_kwargs['activation'] = 'linear'

    q_values = output_layer(units=num_actions, name=f'{name}_q_values', **output_kwargs)(fc_output)
    return q_values


# --- Complete CNN DQN Builder ---
def build_cnn_dqn(input_shape, num_actions, noisy_fc=True, noisy_std_init=0.5,
                  fc_units=512, activation='relu', name="CNNDQN", input_dtype=tf.uint8):
    """Builds a standard CNN DQN using the feature extractor and head."""
    # Feature extractor handles input shape/dtype and normalization
    feature_extractor = build_cnn_feature_extractor(input_shape, activation, name=f"{name}_Features", input_dtype=input_dtype)
    # Head builds the FC layers
    head_output = build_cnn_dqn_head(feature_extractor.output, num_actions, noisy_fc,
                                     noisy_std_init, fc_units, activation, name=f"{name}_Head")
    # Combine into a single model
    model = models.Model(inputs=feature_extractor.input, outputs=head_output, name=name)
    # Summaries printed within component builders
    return model


# --- Actor-Critic Network Builder (Handles 1D/2D/3D Inputs) ---
def build_actor_critic_network(input_shape, num_actions,
                               shared_units=(128,), # Shared layers config
                               actor_units=(64,),   # Actor-specific layers
                               critic_units=(64,),  # Critic-specific layers
                               activation='relu',
                               input_type='vector', # 'vector', 'grid', or 'screen'
                               name="ActorCritic"):
    """
    Builds an Actor-Critic network with shared base layers.
    Handles 1D, 2D, or 3D input types.
    Outputs: [action_logits, state_value]
    """
    input_layer = None
    shared_base_output = None

    # --- Input Layer and Shared Base ---
    if input_type == 'vector':
        if not isinstance(input_shape, tuple) or len(input_shape) != 1:
             raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
        input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input_vector')
        shared_base_output = _create_dense_block(input_layer, shared_units, activation, layers.Dense, {}, f"{name}_shared")
        print(f"\n--- Built ActorCritic Base: MLP (Input: {input_shape}) ---")
    elif input_type in ['grid', 'screen']:
        input_dtype = tf.float32 if input_type == 'grid' else tf.uint8
        cnn_base = build_cnn_feature_extractor(input_shape, activation, f"{name}_shared_CNN", input_dtype)
        input_layer = cnn_base.input
        shared_base_output = cnn_base.output
        print(f"\n--- Built ActorCritic Base: CNN (Input: {cnn_base.input_shape}, Type: {input_type}) ---")
    else:
        raise ValueError(f"Invalid input_type '{input_type}' for {name}.")

    # --- Actor Head (Policy) ---
    actor_output = _create_dense_block(shared_base_output, actor_units, activation, layers.Dense, {}, f"{name}_actor")
    action_logits = layers.Dense(num_actions, activation=None, name=f'{name}_actor_logits')(actor_output) # Logits output

    # --- Critic Head (Value) ---
    critic_output = _create_dense_block(shared_base_output, critic_units, activation, layers.Dense, {}, f"{name}_critic")
    state_value = layers.Dense(1, activation=None, name=f'{name}_critic_value')(critic_output) # Single value output

    # --- Create Model ---
    model = models.Model(inputs=input_layer, outputs=[action_logits, state_value], name=name)
    print(f"--- Completed ActorCritic Network: {name} ---")
    model.summary(line_length=120)
    print("-" * (len(name) + 40))
    return model


# --- Dyna-Q World Model Network Builder (1D State Only) ---
def build_world_model_network(state_size, hidden_units=(64, 64), activation='relu', name="WorldModel"):
    """
    Builds MLP network for Dyna-Q's world model (predicts 1D next state & reward).
    Input: state (1D vector)
    Output: [predicted_next_state (1D vector), predicted_reward (scalar)]
    """
    state_input = layers.Input(shape=(state_size,), dtype=tf.float32, name=f'{name}_state_input')
    hidden_output = _create_dense_block(state_input, hidden_units, activation, layers.Dense, {}, f"{name}_shared")
    predicted_next_state = layers.Dense(state_size, activation=None, name=f'{name}_pred_next_state')(hidden_output)
    predicted_reward = layers.Dense(1, activation=None, name=f'{name}_pred_reward')(hidden_output)
    model = models.Model(inputs=state_input, outputs=[predicted_next_state, predicted_reward], name=name)
    print(f"\n--- Built Network: {name} (Input: {(state_size,)}, Output: State({state_size}), Reward(1)) ---")
    model.summary(line_length=100)
    print("-" * (len(name) + 40))
    return model


# --- ============================================ ---
# --- Existing DQN Variant Builders (1D Vector Input) ---
# --- ============================================ ---

def build_dense_dqn(input_shape, num_actions, hidden_units=(64, 64), activation='relu', name="DenseDQN"):
    """Builds a simple sequential dense network for DQN (1D Vector State)."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    x = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, name)
    q_values = layers.Dense(num_actions, activation='linear', name=f'{name}_q_values')(x)
    model = models.Model(inputs=input_layer, outputs=q_values, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}) ---")
    model.summary(line_length=100)
    print("-" * (len(name) + 30))
    return model

def build_per_dqn(input_shape, num_actions, hidden_units=(64, 32, 16), activation='relu', name="PER_DQN"):
    """Builds a sequential dense network suitable for DQN_PER_Agent (1D Vector State)."""
    # Reuse the dense DQN builder
    return build_dense_dqn(input_shape, num_actions, hidden_units, activation, name)


def build_dueling_dqn(input_shape, num_actions, noisy=False, noisy_std_init=0.5,
                        shared_hidden_units=(128, 128), value_hidden_units=(), advantage_hidden_units=(),
                        activation='relu', name="DuelingDQN"):
    """Builds a Dueling DQN architecture for 1D Vector State."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    LayerType = NoisyLinear if noisy else layers.Dense
    layer_kwargs = {'std_init': noisy_std_init} if noisy else {}

    # Shared Stream
    shared_output = _create_dense_block(input_layer, shared_hidden_units, activation, LayerType, layer_kwargs, f"{name}_shared")

    # Value Stream V(s)
    v = _create_dense_block(shared_output, value_hidden_units, activation, LayerType, layer_kwargs, f"{name}_value")
    value_layer = NoisyLinear if noisy else layers.Dense
    value_kwargs = {'std_init': noisy_std_init} if noisy else {'activation': 'linear'}
    value = value_layer(1, name=f'{name}_value_stream', **value_kwargs)(v)

    # Advantage Stream A(s,a)
    adv = _create_dense_block(shared_output, advantage_hidden_units, activation, LayerType, layer_kwargs, f"{name}_adv")
    advantage_layer = NoisyLinear if noisy else layers.Dense
    adv_kwargs = {'std_init': noisy_std_init} if noisy else {'activation': 'linear'}
    advantages = advantage_layer(num_actions, name=f'{name}_advantage_stream', **adv_kwargs)(adv)

    # Combine Streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    mean_advantage = MeanReducer(axis=1, keepdims=True, name=f'{name}_mean_advantage')(advantages)
    # Use layers.subtract for combining
    q_values = layers.Add(name=f'{name}_combine_q')([value, layers.subtract([advantages, mean_advantage])])

    model = models.Model(inputs=input_layer, outputs=q_values, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}, Noisy={noisy}) ---")
    model.summary(line_length=120)
    print("-" * (len(name) + 40))
    return model


def build_lstm_dueling_dqn(input_shape, num_actions, lstm_units, noisy=False, noisy_std_init=0.5,
                           shared_hidden_units=(64, 64), value_hidden_units=(), advantage_hidden_units=(),
                           activation='relu', name="LSTMDuelingDQN"):
    """Builds a Dueling DQN with an initial LSTM layer for 1D Vector State sequences."""
    if len(input_shape) != 2: # Expects (timesteps, features)
        raise ValueError(f"{name} expects input_shape=(timesteps, features), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    lstm_out = layers.LSTM(lstm_units, activation=activation, return_sequences=False, name=f'{name}_lstm')(input_layer)

    LayerType = NoisyLinear if noisy else layers.Dense
    layer_kwargs = {'std_init': noisy_std_init} if noisy else {}

    # Shared Stream after LSTM
    shared_output = _create_dense_block(lstm_out, shared_hidden_units, activation, LayerType, layer_kwargs, f"{name}_shared")

    # Value Stream V(s)
    v = _create_dense_block(shared_output, value_hidden_units, activation, LayerType, layer_kwargs, f"{name}_value")
    value_layer = NoisyLinear if noisy else layers.Dense
    value_kwargs = {'std_init': noisy_std_init} if noisy else {'activation': 'linear'}
    value = value_layer(1, name=f'{name}_value_stream', **value_kwargs)(v)

    # Advantage Stream A(s,a)
    adv = _create_dense_block(shared_output, advantage_hidden_units, activation, LayerType, layer_kwargs, f"{name}_adv")
    advantage_layer = NoisyLinear if noisy else layers.Dense
    adv_kwargs = {'std_init': noisy_std_init} if noisy else {'activation': 'linear'}
    advantages = advantage_layer(num_actions, name=f'{name}_advantage_stream', **adv_kwargs)(adv)

    # Combine Streams
    mean_advantage = MeanReducer(axis=1, keepdims=True, name=f'{name}_mean_advantage')(advantages)
    q_values = layers.Add(name=f'{name}_combine_q')([value, layers.subtract([advantages, mean_advantage])])

    model = models.Model(inputs=input_layer, outputs=q_values, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}, Noisy={noisy}) ---")
    model.summary(line_length=120)
    print("-" * (len(name) + 40))
    return model


def build_c51_dqn(input_shape, num_actions, num_atoms, v_min, v_max,
                  hidden_units=(64, 64), activation='relu', name="C51DQN"):
    """Builds a C51 network using a dense feature extractor (1D Vector State)."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_feature_input')
    feature_output = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_features")
    # C51 Head
    x = layers.Dense(units=num_actions * num_atoms, activation=None, name=f'{name}_Head_logits')(feature_output)
    reshaped_logits = layers.Reshape((num_actions, num_atoms), name=f'{name}_Head_reshape')(x)
    distribution_output = layers.Softmax(axis=2, name=f'{name}_Head_probs')(reshaped_logits)

    model = models.Model(inputs=input_layer, outputs=distribution_output, name=name)
    # Attach C51 parameters
    model.num_atoms=num_atoms; model.v_min=v_min; model.v_max=v_max
    model.delta_z=(v_max-v_min)/(num_atoms-1) if num_atoms > 1 else 0; model.support=tf.cast(tf.linspace(v_min,v_max,num_atoms),tf.float32)
    print(f"\n--- Built Network: {name} (Input: {input_shape}) ---"); model.summary(line_length=100)
    print(f" C51 Params: Atoms={num_atoms}, Vmin={v_min}, Vmax={v_max}, dz={model.delta_z:.4f}"); print("-"*(len(name)+30))
    return model


def build_bootstrapped_dqn(input_shape, num_actions, num_heads,
                           hidden_units=(64, 64), activation='relu', name="BootstrappedDQN"):
    """Builds a Bootstrapped DQN network with multiple heads (1D Vector State)."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_feature_input')
    feature_output = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_features")
    # Multi-Head Output
    heads = []
    for i in range(num_heads):
        head_q_values = layers.Dense(num_actions, activation='linear', name=f'{name}_Head_q_head_{i+1}')(feature_output)
        heads.append(head_q_values)
    multi_head_output = StackLayer(axis=1, name=f'{name}_Head_stack')(heads)

    model = models.Model(inputs=input_layer, outputs=multi_head_output, name=name)
    model.num_heads = num_heads
    print(f"\n--- Built Network: {name} (Input: {input_shape}) ---"); model.summary(line_length=100)
    print(f" Bootstrap Params: Heads={num_heads}"); print("-" * (len(name) + 30))
    return model


def build_rnd_network(input_shape, output_dim=128, hidden_units=(64, 64), activation='relu', name="RND_Net"):
    """Builds a generic network for RND (Target or Predictor) (1D Vector State)."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    x = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, name)
    output = layers.Dense(output_dim, activation=None, name=f'{name}_output')(x) # Linear output
    model = models.Model(inputs=input_layer, outputs=output, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}) ---"); model.summary(line_length=100); print("-" * (len(name) + 20))
    return model

# --- ================================================== ---
# --- Network Builders for QR-DQN and ICM-DQN ---
# --- ================================================== ---

def build_qr_dqn(input_shape, num_actions, num_quantiles,
                 hidden_units=(64, 64), activation='relu', name="QRDQN"):
    """Builds a QR-DQN network using a dense feature extractor (1D Vector State)."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape (e.g., (state_size,)), got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_feature_input')
    feature_output = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_features")
    # QR-DQN Head - Outputs quantile values directly
    x = layers.Dense(units=num_actions * num_quantiles, activation=None, name=f'{name}_Head_quantiles_flat')(feature_output)
    quantile_output = layers.Reshape((num_actions, num_quantiles), name=f'{name}_Head_reshape')(x)

    model = models.Model(inputs=input_layer, outputs=quantile_output, name=name)
    # Attach QR parameters
    model.num_quantiles = num_quantiles
    model.tau_hat = tf.cast( (tf.range(num_quantiles, dtype=tf.float32) + 0.5) / num_quantiles, dtype=tf.float32)

    print(f"\n--- Built Network: {name} (Input: {input_shape}) ---"); model.summary(line_length=100)
    print(f" QR-DQN Params: Quantiles={num_quantiles}"); print("-"*(len(name)+30))
    return model

def build_icm_encoder(input_shape, feature_dim, hidden_units=(64, 64), activation='relu', name="ICM_Encoder"):
    """Builds the ICM feature encoder network phi(s)."""
    # Assumes vector input_shape = (state_size,) for now
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape, got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    # Could add CNN layers here if input_shape was for images/grids
    x = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_hidden")
    feature_output = layers.Dense(feature_dim, activation=None, name=f'{name}_output')(x) # Linear output features
    model = models.Model(inputs=input_layer, outputs=feature_output, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}, Output Dim: {feature_dim}) ---")
    model.summary(line_length=100); print("-" * (len(name) + 20))
    return model

def build_icm_forward_model(feature_dim, action_size, hidden_units=(64, 64), activation='relu', name="ICM_Forward"):
    """Builds the ICM forward dynamics model f(phi(s), a_one_hot). Predicts phi(s')."""
    feature_input = layers.Input(shape=(feature_dim,), dtype=tf.float32, name=f'{name}_feature_input')
    action_input = layers.Input(shape=(action_size,), dtype=tf.float32, name=f'{name}_action_input') # Expects one-hot action
    # Concatenate features and action
    concat_input = layers.Concatenate(axis=-1, name=f'{name}_concat')([feature_input, action_input])
    x = _create_dense_block(concat_input, hidden_units, activation, layers.Dense, {}, f"{name}_hidden")
    predicted_feature = layers.Dense(feature_dim, activation=None, name=f'{name}_output')(x) # Predict next feature vector
    model = models.Model(inputs=[feature_input, action_input], outputs=predicted_feature, name=name)
    print(f"\n--- Built Network: {name} (Input Dims: Feat={feature_dim}, Act={action_size}, Output Dim: {feature_dim}) ---")
    model.summary(line_length=100); print("-" * (len(name) + 20))
    return model

def build_icm_inverse_model(feature_dim, action_size, hidden_units=(64, 64), activation='relu', name="ICM_Inverse"):
    """Builds the ICM inverse dynamics model g(phi(s), phi(s')). Predicts action logits."""
    feature_input_s = layers.Input(shape=(feature_dim,), dtype=tf.float32, name=f'{name}_feature_s_input')
    feature_input_s_prime = layers.Input(shape=(feature_dim,), dtype=tf.float32, name=f'{name}_feature_s_prime_input')
    # Concatenate features
    concat_input = layers.Concatenate(axis=-1, name=f'{name}_concat')([feature_input_s, feature_input_s_prime])
    x = _create_dense_block(concat_input, hidden_units, activation, layers.Dense, {}, f"{name}_hidden")
    action_logits = layers.Dense(action_size, activation=None, name=f'{name}_output_logits')(x) # Predict action logits
    model = models.Model(inputs=[feature_input_s, feature_input_s_prime], outputs=action_logits, name=name)
    print(f"\n--- Built Network: {name} (Input Dim: {feature_dim}x2, Output Actions: {action_size}) ---")
    model.summary(line_length=100); print("-" * (len(name) + 20))
    return model

# --- ================================================== ---
# --- NEW Network Builders for DDPG/TD3/SAC (Discrete) ---
# --- ================================================== ---

def build_mlp_actor(input_shape, num_actions, hidden_units=(256, 256), activation='relu', name="MLPActor"):
    """Builds a simple MLP Actor network outputting action logits."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape, got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    x = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_hidden")
    # Output logits (linear activation) for discrete actions
    action_logits = layers.Dense(num_actions, activation=None, name=f'{name}_logits')(x)
    model = models.Model(inputs=input_layer, outputs=action_logits, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}, Output: {num_actions} logits) ---")
    model.summary(line_length=100)
    print("-" * (len(name) + 30))
    return model

def build_mlp_critic(input_shape, num_actions, hidden_units=(256, 256), activation='relu', name="MLPCritic"):
    """Builds a simple MLP Critic network outputting Q-values for all actions."""
    if not isinstance(input_shape, tuple) or len(input_shape) != 1:
         raise ValueError(f"{name}: Expected 1D input_shape, got {input_shape}")
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32, name=f'{name}_input')
    x = _create_dense_block(input_layer, hidden_units, activation, layers.Dense, {}, f"{name}_hidden")
    # Output Q-value for each action (linear activation)
    q_values = layers.Dense(num_actions, activation=None, name=f'{name}_q_values')(x)
    model = models.Model(inputs=input_layer, outputs=q_values, name=name)
    print(f"\n--- Built Network: {name} (Input: {input_shape}, Output: {num_actions} Q-values) ---")
    model.summary(line_length=100)
    print("-" * (len(name) + 30))
    return model

# --- END OF FILE project/networks/tf_networks.py ---
