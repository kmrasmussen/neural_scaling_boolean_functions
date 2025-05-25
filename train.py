import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
from functools import partial
import json
from uuid import uuid4
from datetime import datetime
import time # Import the time module
import math # Import Python's math module

print('jax devices', jax.devices())

def to_sci(number):
   return f"{number:.1e}"

# def generate_msp_sample(key, task_interval_length, n_task_intervals): # Keep this for reference or remove if not used elsewhere
#   n_control_bits = int(jnp.log2(n_task_intervals).item())
#   n_task_bits = n_task_intervals * task_interval_length
#   total_bits = n_control_bits + n_task_bits
#   control_and_task_bits = jax.random.randint(key, (total_bits,), minval=0, maxval=2)
#   control_bits = control_and_task_bits[:n_control_bits]
#   powers_of_2 = 2 ** jnp.arange(n_control_bits - 1, -1, -1)
#   control_base10 = (control_bits * powers_of_2).sum()
#   selected_task_interval_begin_idx = n_control_bits+control_base10*task_interval_length
#   selected_task_bits = jax.lax.dynamic_slice(
#       control_and_task_bits,
#       (selected_task_interval_begin_idx,),
#       (task_interval_length,)
#   )
#   selected_task_parity = selected_task_bits.sum() % 2
#   return control_and_task_bits, selected_task_parity

@partial(jax.jit, static_argnames=("task_interval_length", "n_task_intervals", "batch_size"))
def generate_msp_batch(key, task_interval_length, n_task_intervals, batch_size):
    # n_task_intervals is a static argument, so it's a concrete Python int here.
    # Perform calculations that need to result in Python ints using Python math.
    n_control_bits = int(math.log2(n_task_intervals)) # Use math.log2 for Python int
    n_task_bits = n_task_intervals * task_interval_length
    total_bits = n_control_bits + n_task_bits
    
    # powers_of_2 can be a JAX array
    powers_of_2 = 2 ** jnp.arange(n_control_bits - 1, -1, -1)

    # Generate all random bits for the entire batch at once
    # Shape: (batch_size, total_bits)
    all_control_and_task_bits = jax.random.randint(key, (batch_size, total_bits), 0, 2)

    # Extract control bits for all samples
    # Shape: (batch_size, n_control_bits)
    all_control_bits = all_control_and_task_bits[:, :n_control_bits]

    # Calculate control_base10 for all samples using matrix multiplication
    # Shape: (batch_size,)
    all_control_base10 = jnp.dot(all_control_bits, powers_of_2)
    
    # Calculate selected_task_interval_begin_idx for all samples
    # Shape: (batch_size,)
    # n_control_bits and task_interval_length are Python ints here, so this is fine.
    all_selected_task_interval_begin_idx = n_control_bits + all_control_base10 * task_interval_length

    # Define the per-sample operation for vmap
    def get_selected_parity(single_control_and_task_bits, single_selected_task_interval_begin_idx):
        # single_control_and_task_bits has shape (total_bits,)
        # single_selected_task_interval_begin_idx is a JAX scalar (0-D array)
        selected_task_bits = jax.lax.dynamic_slice(
            single_control_and_task_bits,
            (single_selected_task_interval_begin_idx,), 
            (task_interval_length,) # task_interval_length is static, so it's a Python int
        )
        return selected_task_bits.sum() % 2

    # Vmap this function. It will iterate over the 0-th axis of both inputs.
    all_selected_task_parities = jax.vmap(get_selected_parity)(
        all_control_and_task_bits, all_selected_task_interval_begin_idx
    )
    # all_selected_task_parities will have shape (batch_size,)

    return all_control_and_task_bits, all_selected_task_parities

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
batch_size = 1028
task_interval_length = 4
n_task_intervals = 8
base_train_key = jax.random.key(42) # Use a base key

learning_rate = 1e-4
loss_epsilon = 0.05 # set lower when ready, e.g. 0.01 for ~6.6 bits
ema_beta = 0.01

# Define the different hidden dimension configurations to try
# Each element in the list is a tuple for hidden_dims
configurations_to_try = [
  (128,128,128),
  (256,256,256),
  (512,512,512),
  (1024,1024,1024),
  (4096,4096),
  (1300,1300),
  (1024,1024),
  (700,700),
  (512,512),
  (256,256),
  (128,128),
  (4096)
]

steps_jsonl_file = 'experiment_runs.jsonl' # Single log file for all runs
last_log_item = None
# Main loop for different configurations
for i, current_model_hidden_dims in enumerate(configurations_to_try):
    print(f"\n--- Starting Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")
    
    run_start_time = time.time() # Start timer for the run
    
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
    print('size of input space', to_sci(input_space_size))
    
    # Initialize model parameters
    init_key, _ = jax.random.split(run_key) # Use a key derived from the run_key
    params = model.init(init_key, dummy_batch_x)['params']

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print('total model params', to_sci(total_params))
    
    forward_flops = model.analyze_forward_flops(dummy_batch_x.shape[-1])
    backward_flops = 2 * forward_flops # Standard approximation
    forward_backward_flops = forward_flops + backward_flops
    batch_flops = batch_size * forward_backward_flops # FLOPs for one batch step
    print('per batch forward-backward-flops (batch_flops)', to_sci(batch_flops))


    # Reset EMA and cumulative metrics for each run
    ema_accuracy = 0.5 
    ema_loss = jnp.log(2) 
    cumulative_surprisal = 0.0
    ema_flops_per_second = 0.0 # Initialize EMA for FLOPs/sec

    run_uuid = str(uuid4())

    n_nonimproving_steps = 0
    n_improving_steps = 0
    #continue
    with open(steps_jsonl_file, 'a') as f_log:
        for batch_step in range(num_batch_steps):
            # batch_loop_start_time = time.time() # You can keep this for overall loop time if you want

            data_gen_start_time = time.time()
            epoch_key, train_key = jax.random.split(train_key)
            batch_x_epoch, batch_y_epoch = generate_msp_batch(
                epoch_key,
                task_interval_length,
                n_task_intervals,
                batch_size)
            # Ensure data generation is complete if it involves JAX operations that might be async
            # (though jax.random and slicing are often synchronous or JIT-compiled effectively)
            # For safety, you could block on the generated data if you suspect it's JAX-based and async
            # batch_x_epoch.block_until_ready() 
            # batch_y_epoch.block_until_ready()
            data_gen_end_time = time.time()
            data_gen_time = data_gen_end_time - data_gen_start_time
            
            # Time the train_step call
            train_step_start_time = time.time()
            params, opt_state, loss, accuracy, losses = train_step(
                params,
                opt_state,
                batch_x_epoch,
                batch_y_epoch,
                model.apply,
                optimizer
            )
            jax.tree_util.tree_leaves(params)[0].block_until_ready()
            train_step_end_time = time.time()
            
            actual_train_step_time = train_step_end_time - train_step_start_time # Renamed for clarity
            
            # Calculate FLOPs/sec for the current batch using actual_train_step_time
            current_flops_per_second = 0
            if actual_train_step_time > 0:
                current_flops_per_second = batch_flops / actual_train_step_time
            else: 
                current_flops_per_second = float('inf')


            loss = loss/jnp.log(2) # convert to units bits
            losses = losses/jnp.log(2)

            cumulative_bits = batch_size*(batch_step+1) 
            cumulative_surprisal += losses.sum()

            if batch_step == 0: 
                ema_loss = loss 
                ema_accuracy = accuracy
                ema_flops_per_second = current_flops_per_second # Initialize with first measurement
            else:
                ema_accuracy = (1-ema_beta)*ema_accuracy + ema_beta*accuracy
                ema_loss = (1-ema_beta)*ema_loss + ema_beta*loss
                ema_flops_per_second = (1-ema_beta)*ema_flops_per_second + ema_beta*current_flops_per_second
            
            if batch_step % 100 == 0:
                # Total time elapsed for the current run up to this logging step
                current_run_time = time.time() 
                cumseconds = current_run_time - run_start_time

                log_item = {
                    'run_config_index': i,
                    'model_hidden_dims': current_model_hidden_dims,
                    'emaacc': ema_accuracy.item(),
                    'emaloss': ema_loss.item(),
                    'cumulsurp': cumulative_surprisal.item(),
                    'cumulbits': cumulative_bits,
                    'batch_size': batch_size,
                    'batch_step': batch_step,
                    'lr': learning_rate,
                    'task_interval_length': task_interval_length,
                    'n_task_intervals': n_task_intervals,
                    'run_uuid': run_uuid,
                    'total_params': total_params,
                    'batch_flops': batch_flops,
                    'data_generation_time': data_gen_time, # Log this
                    'batch_time': actual_train_step_time, # This is the time for FLOPs calculation
                    'flops_per_second': to_sci(current_flops_per_second), 
                    'ema_flops_per_second': to_sci(ema_flops_per_second), 
                    'forward_backward_flops': forward_backward_flops,
                    'cumseconds': cumseconds # Cumulative seconds for this run
                }
                #print(log_item)
                print('emaloss', ema_loss)
                f_log.write(json.dumps(log_item) + '\n')
                f_log.flush() 

                if ema_loss < loss_epsilon:
                    print(f"Run {i+1} with {current_model_hidden_dims}: EMA loss {ema_loss.item():.4f} < {loss_epsilon}. Stopping.")
                    break 
                if batch_step > 10000:
                  if last_log_item['emaloss'] < log_item['emaloss']:
                    n_nonimproving_steps += 1
                    if n_nonimproving_steps > 100:
                      print('n nonimproving steps exceeded 10 ')
                      break
                  if log_item['emaloss'] < last_log_item['emaloss']:
                    n_improving_steps += 1
                    if n_improving_steps > 200:
                      print('saw steady improvement, resetting early stopper')
                      n_nonimproving_steps = 0
                      n_improving_steps = 0
                last_log_item = log_item
                
    
    print(f"--- Finished Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")

# The file is automatically closed when the 'with' block exits after the loop
# ... any post-processing or summary across all runs can go here