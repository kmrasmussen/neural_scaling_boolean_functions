import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
from functools import partial
import json
from uuid import uuid4
from datetime import datetime

def to_sci(number):
   return f"{number:.1e}"

def generate_msp_sample(key, task_interval_length, n_task_intervals):
  n_control_bits = int(jnp.log2(n_task_intervals).item())
  n_task_bits = n_task_intervals * task_interval_length
  total_bits = n_control_bits + n_task_bits
  control_and_task_bits = jax.random.randint(key, (total_bits), minval=0, maxval=2)
  control_bits = control_and_task_bits[:n_control_bits]
  powers_of_2 = 2 ** jnp.arange(n_control_bits - 1, -1, -1)
  control_base10 = (control_bits * powers_of_2).sum()
  selected_task_interval_begin_idx = n_control_bits+control_base10*task_interval_length
  selected_task_bits = jax.lax.dynamic_slice(
      control_and_task_bits,
      (selected_task_interval_begin_idx,),
      (task_interval_length,)
  )
  selected_task_parity = selected_task_bits.sum() % 2
  return control_and_task_bits, selected_task_parity

def generate_msp_batch(key, task_interval_length, n_task_intervals, batch_size):
  keys = jax.random.split(key, batch_size)
  batch_fn = jax.vmap(generate_msp_sample, in_axes=(0,None,None))
  batch = batch_fn(keys, task_interval_length, n_task_intervals)
  return batch

class MLP(nn.Module):
  hidden_dims: tuple

  @nn.compact
  def __call__(self, x):
    for dim in self.hidden_dims:
      x = nn.Dense(dim)(x)
      x = nn.relu(x)
    x = nn.Dense(1)(x)
    return x.squeeze(-1)
  
  def analyze_forward_flops(self, example_input_feature_dim: int):
    estimated_forward_flops = 0
    current_features = example_input_feature_dim
    for out_features in self.hidden_dims:
        # Dense layer: Wx + b
        # Matmul (Wx): approx 2 * current_features * out_features FLOPs
        # Bias add (+b): out_features FLOPs
        estimated_forward_flops += (2 * current_features * out_features) + out_features
        # ReLU activation: out_features FLOPs (each element compared to 0)
        estimated_forward_flops += out_features
        current_features = out_features
    
    # Final Dense layer (output layer)
    final_out_features = 1 # As defined in __call__
    # Matmul (Wx): approx 2 * current_features * final_out_features FLOPs
    # Bias add (+b): final_out_features FLOPs
    estimated_forward_flops += (2 * current_features * final_out_features) + final_out_features
    # No ReLU after the final layer in this MLP's __call__

    return estimated_forward_flops

@jax.jit
def binary_cross_entropy_loss(logits, labels):
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

def _train_step_impl(params,
               opt_state, 
               batch_x, 
               batch_y, 
               model_apply_fn, 
               current_optimizer):
  def loss_fn(params):
    logits = model_apply_fn({'params': params}, batch_x)
    losses = optax.sigmoid_binary_cross_entropy(logits, batch_y)
    loss = binary_cross_entropy_loss(logits, batch_y)
    return loss, (logits, losses)

  (loss, (logits, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = current_optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  accuracy = jnp.mean((logits > 0) == batch_y)
  return params, opt_state, loss, accuracy, losses

# %%
train_step = jax.jit(_train_step_impl, static_argnames=['model_apply_fn', 'current_optimizer'])
# %%
num_batch_steps = 1000000
batch_size = 32
task_interval_length = 4
n_task_intervals = 4
base_train_key = jax.random.key(42) # Use a base key

learning_rate = 1e-4
loss_epsilon = 0.995 # set lower when ready, e.g. 0.01 for ~6.6 bits
ema_beta = 0.01

# Define the different hidden dimension configurations to try
# Each element in the list is a tuple for hidden_dims
configurations_to_try = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (256, 256, 256) # Example of a 3-layer MLP
]

steps_jsonl_file = 'experiment_runs.jsonl' # Single log file for all runs

# Main loop for different configurations
for i, current_model_hidden_dims in enumerate(configurations_to_try):
    print(f"\n--- Starting Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")
    
    # Ensure each run uses a different key for reproducibility if needed
    run_key, base_train_key = jax.random.split(base_train_key)
    train_key = run_key # Use this key for the current run's batch generation

    model = MLP(hidden_dims=current_model_hidden_dims)
    
    # Use a consistent key for dummy batch generation if its content doesn't need to vary per run
    # Or use a split from run_key if it should vary
    dummy_key, _ = jax.random.split(jax.random.key(0)) 
    dummy_batch_x, dummy_batch_y = generate_msp_batch(
      dummy_key, # Consistent dummy batch
      task_interval_length,
      n_task_intervals,
      1)

    x_seq_length = dummy_batch_x.shape[-1]
    input_space_size = 2 ** x_seq_length
    print('size of input space', input_space_size)
    
    # Initialize model parameters
    init_key, _ = jax.random.split(run_key) # Use a key derived from the run_key
    params = model.init(init_key, dummy_batch_x)['params']

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print('total model params', total_params)
    
    # Analyze FLOPs for the current model configuration
    # Note: analyze_forward_flops is a method of an instance, so create one.
    # Or, if it doesn't depend on initialized params, you can call it on the 'model' instance.
    forward_flops = model.analyze_forward_flops(dummy_batch_x.shape[-1])
    print('forward flops', forward_flops)
    backward_flops = 2 * forward_flops
    # total_flops = forward_flops * backward_flops # This seems like a typo, usually it's sum for total per step
    batch_forward_backward_flops = batch_size * (forward_flops + backward_flops)
    print('per batch forward-backward-flops', batch_forward_backward_flops)

    # Reset EMA and cumulative metrics for each run
    ema_accuracy = 0.5 
    ema_loss = jnp.log(2) # Initialize to a high value (e.g. random guessing in bits)
    cumulative_surprisal = 0.0

    run_uuid = str(uuid4())

    with open(steps_jsonl_file, 'a') as f_log:
        for batch_step in range(num_batch_steps):
            epoch_key, train_key = jax.random.split(train_key)
            batch_x_epoch, batch_y_epoch = generate_msp_batch(
                epoch_key,
                task_interval_length,
                n_task_intervals,
                batch_size)
            
            params, opt_state, loss, accuracy, losses = train_step(
                params,
                opt_state,
                batch_x_epoch,
                batch_y_epoch,
                model.apply,
                optimizer
            )
            loss = loss/jnp.log(2) # convert to units bits
            losses = losses/jnp.log(2)

            cumulative_bits = batch_size*(batch_step+1) # Total bits processed so far in this run
            cumulative_surprisal += losses.sum()

            if batch_step == 0: # Initialize EMA on the first step
                ema_loss = loss 
                ema_accuracy = accuracy
            else:
                ema_accuracy = (1-ema_beta)*ema_accuracy + ema_beta*accuracy
                ema_loss = (1-ema_beta)*ema_loss + ema_beta*loss
            
            if batch_step % 1000 == 0:
                log_item = {
                    'run_config_index': i,
                    'model_hidden_dims': current_model_hidden_dims, # Log current config
                    'emaacc': ema_accuracy.item(),
                    'emaloss': ema_loss.item(),
                    'cumulsurp': cumulative_surprisal.item(),
                    'cumulbits': cumulative_bits, # This is cumulative bits for the current run
                    'batch_size': batch_size,
                    'batch_step': batch_step,
                    'lr': learning_rate,
                    'task_interval_length': task_interval_length,
                    'n_task_intervals': n_task_intervals,
                    'run_uuid': run_uuid, # UUID for this specific run
                    'total_params': total_params,
                    'batch_forward_backward_flops': batch_forward_backward_flops
                }
                print(log_item) # Optional: print to console
                f_log.write(json.dumps(log_item) + '\n')
                f_log.flush() 

                if ema_loss < loss_epsilon:
                    print(f"Run {i+1} with {current_model_hidden_dims}: EMA loss {ema_loss.item():.4f} < {loss_epsilon}. Stopping.")
                    break # Stop training for this configuration
    
    print(f"--- Finished Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")

# The file is automatically closed when the 'with' block exits after the loop
# ... any post-processing or summary across all runs can go here