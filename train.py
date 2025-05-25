import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
from functools import partial
import json
from uuid import uuid4
from datetime import datetime
import time
import math

def to_sci(number):
   return f"{number:.1e}"

@partial(jax.jit, static_argnames=("task_interval_length", "n_task_intervals", "batch_size"))
def generate_msp_batch(key, task_interval_length, n_task_intervals, batch_size):
    n_control_bits = int(math.log2(n_task_intervals)) # Use math.log2 for Python int
    n_task_bits = n_task_intervals * task_interval_length
    total_bits = n_control_bits + n_task_bits
    powers_of_2 = 2 ** jnp.arange(n_control_bits - 1, -1, -1)
    all_control_and_task_bits = jax.random.randint(key, (batch_size, total_bits), 0, 2)
    all_control_bits = all_control_and_task_bits[:, :n_control_bits]
    all_control_base10 = jnp.dot(all_control_bits, powers_of_2)
    all_selected_task_interval_begin_idx = n_control_bits + all_control_base10 * task_interval_length
    def get_selected_parity(single_control_and_task_bits, single_selected_task_interval_begin_idx):
        selected_task_bits = jax.lax.dynamic_slice(
            single_control_and_task_bits,
            (single_selected_task_interval_begin_idx,), 
            (task_interval_length,)
        )
        return selected_task_bits.sum() % 2
    all_selected_task_parities = jax.vmap(get_selected_parity)(
        all_control_and_task_bits, all_selected_task_interval_begin_idx
    )
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
        estimated_forward_flops += (2 * current_features * out_features) + out_features
        estimated_forward_flops += out_features
        current_features = out_features
    
    final_out_features = 1
    estimated_forward_flops += (2 * current_features * final_out_features) + final_out_features

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
    losses = optax.sigmoid_binary_cross_entropy(logits, batch_y)/jnp.log(2) # unit bits
    loss = jnp.mean(losses) # unit bits
    return loss, (logits, losses)

  (loss, (logits, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = current_optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  accuracy = jnp.mean((logits > 0) == batch_y)
  return params, opt_state, loss, accuracy, losses
train_step = jax.jit(_train_step_impl, static_argnames=['model_apply_fn', 'current_optimizer'])

# Configuration
num_batch_steps = 1000000
batch_size = 1028
task_interval_length = 4
n_task_intervals = 8
base_train_key = jax.random.key(42)

learning_rate = 1e-4
loss_epsilon = 0.05
ema_beta = 0.01

max_width = 4096
min_width = 32
num_widths = 20
log_spaced_widths = jnp.geomspace(
  min_width,
  max_width+1,
  num_widths,
  dtype=jnp.int32
)
configurations_to_try = [
  (int(x), int(x))
  for x in log_spaced_widths
]

print(configurations_to_try)

steps_jsonl_file = 'experiment_runs.jsonl' # Single log file for all runs
# Main loop for different configurations
for i, current_model_hidden_dims in enumerate(configurations_to_try):
  print(f"\n--- Starting Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")
  
  run_start_time = time.time() # Start timer for the run
  
  run_key, base_train_key = jax.random.split(base_train_key)
  train_key = run_key # Use this key for the current run's batch generation

  model = MLP(hidden_dims=current_model_hidden_dims)
  
  dummy_key, _ = jax.random.split(jax.random.key(0)) 
  dummy_batch_x, dummy_batch_y = generate_msp_batch(
    dummy_key, # Consistent dummy batch
    task_interval_length,
    n_task_intervals,
    1)

  x_seq_length = dummy_batch_x.shape[-1]
  input_space_size = 2 ** x_seq_length
  print('size of input space', to_sci(input_space_size))
  
  init_key, _ = jax.random.split(run_key)
  params = model.init(init_key, dummy_batch_x)['params']

  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)

  total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print('total model params', to_sci(total_params))
  
  forward_flops = model.analyze_forward_flops(dummy_batch_x.shape[-1])
  backward_flops = 2 * forward_flops
  forward_backward_flops = forward_flops + backward_flops
  batch_flops = batch_size * forward_backward_flops
  print('forward-backward-flops per sample', to_sci(forward_backward_flops))

  last_log_item = None
  ema_accuracy = 0.5 
  ema_loss = jnp.log(2) 
  cumulative_surprisal = 0.0
  ema_flops_per_second = 0.0
  run_uuid = str(uuid4())
  n_nonimproving_steps = 0
  n_improving_steps = 0
  #continue
  with open(steps_jsonl_file, 'a') as f_log:
    for batch_step in range(num_batch_steps):
      # Generate data
      data_gen_start_time = time.time()
      epoch_key, train_key = jax.random.split(train_key)
      batch_x_epoch, batch_y_epoch = generate_msp_batch(
          epoch_key,
          task_interval_length,
          n_task_intervals,
          batch_size)
      data_gen_end_time = time.time()
      data_gen_time = data_gen_end_time - data_gen_start_time

      # Train step
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
      
      actual_train_step_time = train_step_end_time - train_step_start_time
      
      current_flops_per_second = 0
      current_flops_per_second = batch_flops / actual_train_step_time

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
          'data_generation_time': data_gen_time,
          'batch_time': actual_train_step_time,
          'flops_per_second': to_sci(current_flops_per_second), 
          'ema_flops_per_second': to_sci(ema_flops_per_second), 
          'forward_backward_flops': forward_backward_flops,
          'cumseconds': cumseconds
        }
        #print(log_item)
        print('emaloss', ema_loss)
        f_log.write(json.dumps(log_item) + '\n')
        f_log.flush() 

        if ema_loss < loss_epsilon:
            print(f"Run {i+1} with {current_model_hidden_dims}: EMA loss {ema_loss.item():.4f} < {loss_epsilon}. Stopping.")
            break 

        # Some kind of early stopping
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