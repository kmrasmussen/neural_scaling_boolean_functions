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
import random_circuit_sampler

def to_sci(number):
   return f"{number:.1e}"

@partial(jax.jit, static_argnames=("task_interval_length", "n_task_intervals", "batch_size"))
def generate_msp_batch(key, task_interval_length, n_task_intervals, batch_size):
    n_control_bits = int(math.log2(n_task_intervals)) 
    n_task_bits = n_task_intervals * task_interval_length
    total_bits = n_control_bits + n_task_bits
    powers_of_2 = 2 ** jnp.arange(n_control_bits - 1, -1, -1)
    all_control_and_task_bits = jax.random.randint(key, (batch_size, total_bits), 0, 2, dtype=jnp.int32)
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
    return all_control_and_task_bits, all_selected_task_parities.astype(jnp.int32)

class MLP(nn.Module):
  hidden_dims: tuple

  @nn.compact
  def __call__(self, x):
    x = x.astype(jnp.float32) 
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
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels.astype(jnp.float32)))

def _train_step_impl(params,
               opt_state, 
               batch_x, 
               batch_y, 
               model_apply_fn, 
               current_optimizer):
  def loss_fn(params):
    logits = model_apply_fn({'params': params}, batch_x)
    losses = optax.sigmoid_binary_cross_entropy(logits, batch_y.astype(jnp.float32))/jnp.log(2) 
    loss = jnp.mean(losses) 
    return loss, (logits, losses)

  (loss, (logits, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = current_optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  accuracy = jnp.mean((logits > 0) == batch_y.astype(jnp.bool_)) 
  return params, opt_state, loss, accuracy, losses
train_step = jax.jit(_train_step_impl, static_argnames=['model_apply_fn', 'current_optimizer'])

# Configuration
SAMPLER_TYPE = "circuit"  

task_interval_length = 5
n_task_intervals = 8 

CIRCUIT_N_INPUTS = 256      
CIRCUIT_N_LAYERS = 8       
CIRCUIT_GATES_PER_LAYER = 256
CIRCUIT_N_GATE_TYPES = 4    

num_batch_steps = 100000 
batch_size = 1028        
base_train_key = jax.random.key(42)

learning_rate = 1e-4
loss_epsilon = 0.05
ema_beta = 0.01

max_width = 8000
min_width = 32
num_widths = 20 
log_spaced_widths = jnp.geomspace(
  min_width,
  max_width+1, 
  num_widths,
  dtype=jnp.int32
)
configurations_to_try = [(1000,1000)] #[(int(x), int(x))for x in log_spaced_widths]

print("Configurations to try:", configurations_to_try)
steps_jsonl_file = 'experiment_runs.jsonl' 

# --- GLOBAL SAMPLER SETUP (Done ONCE for all model configurations) ---
sampler_setup_key, base_train_key = jax.random.split(base_train_key) # Key for one-time sampler setup

current_sampler_fn = None
current_sampler_spec = None
current_x_seq_length = None

if SAMPLER_TYPE == "msp":
  print(f"Global Sampler: MSP with task_interval_length={task_interval_length}, n_task_intervals={n_task_intervals}")
  n_control_bits = int(math.log2(n_task_intervals))
  n_task_bits = n_task_intervals * task_interval_length
  current_x_seq_length = n_control_bits + n_task_bits
  
  def msp_sampler_adapter(key, spec, b_size): 
      til, nti = spec
      return generate_msp_batch(key, til, nti, b_size)
  current_sampler_fn = msp_sampler_adapter
  current_sampler_spec = (task_interval_length, n_task_intervals)

elif SAMPLER_TYPE == "circuit":
  print(f"Global Sampler: Circuit with N_inputs={CIRCUIT_N_INPUTS}, N_layers={CIRCUIT_N_LAYERS}, Gates_p_layer={CIRCUIT_GATES_PER_LAYER}")
  current_x_seq_length = CIRCUIT_N_INPUTS
  
  # Generate the circuit specification ONCE using sampler_setup_key
  circuit_spec_actual = random_circuit_sampler.generate_random_circuit_spec(
      sampler_setup_key, # Use the dedicated key
      CIRCUIT_N_INPUTS,
      CIRCUIT_N_LAYERS,
      CIRCUIT_GATES_PER_LAYER,
      CIRCUIT_N_GATE_TYPES
  )
  current_sampler_fn = random_circuit_sampler.create_circuit_sampler(
      num_inputs=CIRCUIT_N_INPUTS,
      num_layers=CIRCUIT_N_LAYERS,
      gates_per_layer=CIRCUIT_GATES_PER_LAYER,
      num_gate_types=CIRCUIT_N_GATE_TYPES
  )
  current_sampler_spec = circuit_spec_actual # This fixed spec will be used for all runs
else:
  raise ValueError(f"Unknown SAMPLER_TYPE: {SAMPLER_TYPE}")

# --- END GLOBAL SAMPLER SETUP ---

# --- Generate a single dummy batch using the global sampler config ---
dummy_key_global = jax.random.key(0) # Consistent dummy key
dummy_batch_x_global, _ = current_sampler_fn(dummy_key_global, current_sampler_spec, 1)
# current_x_seq_length is already set from global setup
input_space_size = 2 ** current_x_seq_length
print('Global size of input space:', to_sci(input_space_size))
print('Global input sequence length (x_seq_length):', current_x_seq_length)
# --- End dummy batch generation ---


for i, current_model_hidden_dims in enumerate(configurations_to_try):
  print(f"\n--- Starting Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")
  
  run_start_time = time.time() 
  
  # Each run gets its own key sequence, derived from the (potentially updated) base_train_key
  run_key, base_train_key = jax.random.split(base_train_key)
  train_key_for_run = run_key # This key will be used for this specific run's batch generation and init

  model = MLP(hidden_dims=current_model_hidden_dims)
  
  # Use the globally generated dummy_batch_x_global for model initialization
  init_key, train_key_for_run = jax.random.split(train_key_for_run) 
  params = model.init(init_key, dummy_batch_x_global)['params']

  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)

  total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print('Total model params:', to_sci(total_params))
  
  forward_flops = model.analyze_forward_flops(current_x_seq_length) 
  backward_flops = 2 * forward_flops
  forward_backward_flops = forward_flops + backward_flops
  batch_flops = batch_size * forward_backward_flops 
  print('Forward-backward-flops per sample:', to_sci(forward_backward_flops))

  last_log_item = None
  ema_accuracy = 0.5 
  ema_loss = jnp.log(2) 
  cumulative_surprisal = 0.0
  ema_flops_per_second = 0.0
  run_uuid = str(uuid4())
  n_nonimproving_steps = 0
  n_improving_steps = 0

  with open(steps_jsonl_file, 'a') as f_log:
    for batch_step in range(num_batch_steps):
      data_gen_start_time = time.time()
      # Use train_key_for_run for this run's data generation
      epoch_key, train_key_for_run = jax.random.split(train_key_for_run)
      
      batch_x_epoch, batch_y_epoch = current_sampler_fn(
          epoch_key,
          current_sampler_spec, # Use the globally defined spec
          batch_size 
      )
      data_gen_end_time = time.time()
      data_gen_time = data_gen_end_time - data_gen_start_time

      train_step_start_time = time.time()
      params, opt_state, loss, accuracy, losses = train_step(
        params,
        opt_state,
        batch_x_epoch,
        batch_y_epoch,
        model.apply,
        optimizer
      )
      if jax.tree_util.tree_leaves(params): 
          jax.tree_util.tree_leaves(params)[0].block_until_ready()
      else: 
          loss.block_until_ready()

      train_step_end_time = time.time()
      
      actual_train_step_time = train_step_end_time - train_step_start_time
      
      current_flops_per_second = 0
      if actual_train_step_time > 0: 
          current_flops_per_second = batch_flops / actual_train_step_time

      cumulative_bits = batch_size*(batch_step+1) 
      cumulative_surprisal += losses.sum()

      if batch_step == 0: 
        ema_loss = loss 
        ema_accuracy = accuracy
        ema_flops_per_second = current_flops_per_second 
      else:
        ema_accuracy = (1-ema_beta)*ema_accuracy + ema_beta*accuracy
        ema_loss = (1-ema_beta)*ema_loss + ema_beta*loss
        ema_flops_per_second = (1-ema_beta)*ema_flops_per_second + ema_beta*current_flops_per_second
      
      if batch_step % 100 == 0 or batch_step == num_batch_steps -1 : 
        current_run_time = time.time() 
        cumseconds = current_run_time - run_start_time

        log_item = {
          'run_config_index': i,
          'model_hidden_dims': current_model_hidden_dims,
          'sampler_type': SAMPLER_TYPE, 
          'emaacc': ema_accuracy.item(),
          'emaloss': ema_loss.item(),
          'cumulsurp': cumulative_surprisal.item(),
          'cumulbits': cumulative_bits,
          'batch_size': batch_size,
          'batch_step': batch_step,
          'lr': learning_rate,
          'run_uuid': run_uuid,
          'total_params': total_params,
          'batch_flops': batch_flops,
          'data_generation_time': data_gen_time,
          'batch_time': actual_train_step_time,
          'flops_per_second': to_sci(current_flops_per_second), 
          'ema_flops_per_second': to_sci(ema_flops_per_second), 
          'forward_backward_flops': forward_backward_flops,
          'cumseconds': cumseconds,
          'x_seq_length': current_x_seq_length
        }
        
        if SAMPLER_TYPE == "msp":
            log_item['task_interval_length'] = task_interval_length
            log_item['n_task_intervals'] = n_task_intervals
        elif SAMPLER_TYPE == "circuit":
            log_item['circuit_n_inputs'] = CIRCUIT_N_INPUTS
            log_item['circuit_n_layers'] = CIRCUIT_N_LAYERS
            log_item['circuit_gates_per_layer'] = CIRCUIT_GATES_PER_LAYER

        print(f"Step: {batch_step}, EMA Loss: {ema_loss.item():.4f}, EMA Acc: {ema_accuracy.item():.4f}, EMA FPS: {to_sci(ema_flops_per_second)}")
        f_log.write(json.dumps(log_item) + '\n')
        f_log.flush() 

        if ema_loss < loss_epsilon:
            print(f"Run {i+1} with {current_model_hidden_dims}: EMA loss {ema_loss.item():.4f} < {loss_epsilon}. Stopping.")
            break 
        
        if batch_step > 10000 and last_log_item is not None: 
          if last_log_item['emaloss'] < log_item['emaloss']: 
            n_nonimproving_steps += 1
            if n_nonimproving_steps > 200: 
              print(f'Early stopping: EMA loss has not improved for {n_nonimproving_steps} log intervals.')
              break
          elif log_item['emaloss'] < last_log_item['emaloss']: 
            n_improving_steps +=1
            if last_log_item['emaloss'] - log_item['emaloss'] > 0.001: 
                 n_nonimproving_steps = 0 
            if n_improving_steps > 400: 
              n_nonimproving_steps = 0
              n_improving_steps = 0
        last_log_item = log_item
        
  print(f"--- Finished Run {i+1}/{len(configurations_to_try)} with Hidden Dims: {current_model_hidden_dims} ---")