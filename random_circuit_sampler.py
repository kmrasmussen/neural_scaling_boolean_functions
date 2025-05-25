import jax
import jax.numpy as jnp
from functools import partial
import time # Add this import

# --- 1. Gate Logic ---
def single_gate_logic_batch(gate_type: int, operand1: jnp.ndarray, operand2: jnp.ndarray) -> jnp.ndarray:
    """
    Applies gate logic for a single gate type to a batch of operands.
    operand1, operand2: shape (batch_size,)
    Returns: shape (batch_size,)
    """
    # Example: 0: AND, 1: OR, 2: XOR, 3: NOT(operand1)
    return jax.lax.switch(
        gate_type,
        [
            lambda op1, op2: jnp.logical_and(op1, op2),
            lambda op1, op2: jnp.logical_or(op1, op2),
            lambda op1, op2: jnp.logical_xor(op1, op2),
            lambda op1, op2: jnp.logical_not(op1), # op2 is ignored
        ],
        operand1,
        operand2,
    )

# --- 2. Generate Circuit Specification ---
def generate_random_circuit_spec(key, num_inputs, num_layers, gates_per_layer, num_gate_types):
    keys = jax.random.split(key, num_layers * 2 + 1)
    gate_types_matrix = jax.random.randint(keys[0], (num_layers, gates_per_layer), 0, num_gate_types)
    
    all_conns1, all_conns2 = [], []

    # Layer 0 connections (from primary inputs)
    conns1_l0 = jax.random.randint(keys[1], (gates_per_layer,), 0, num_inputs)
    conns2_l0 = jax.random.randint(keys[2], (gates_per_layer,), 0, num_inputs)
    all_conns1.append(conns1_l0)
    all_conns2.append(conns2_l0)

    # Subsequent layer connections (from previous layer's gates)
    for i in range(1, num_layers):
        conns1_li = jax.random.randint(keys[2*i+1], (gates_per_layer,), 0, gates_per_layer)
        conns2_li = jax.random.randint(keys[2*i+2], (gates_per_layer,), 0, gates_per_layer)
        all_conns1.append(conns1_li)
        all_conns2.append(conns2_li)
        
    connections_input1 = jnp.stack(all_conns1)
    connections_input2 = jnp.stack(all_conns2)
    
    return gate_types_matrix, connections_input1, connections_input2

# --- 3. Evaluate Circuit ---
# Note: apply_gate_fn, num_layers, gates_per_layer should be static for JIT
def _evaluate_circuit_batch_impl(circuit_spec, batch_x, apply_gate_fn, num_layers, gates_per_layer):
    gate_types_matrix, connections_input1, connections_input2 = circuit_spec
    
    current_layer_input_values = batch_x.astype(jnp.bool_) # Ensure boolean inputs

    for l_idx in range(num_layers):
        gate_types_l = gate_types_matrix[l_idx, :]    # (gates_per_layer,)
        conns1_l = connections_input1[l_idx, :]      # (gates_per_layer,)
        conns2_l = connections_input2[l_idx, :]      # (gates_per_layer,)

        # Gather inputs for this layer's gates
        # Input source is batch_x for l_idx=0, else previous layer's outputs
        source_values = batch_x if l_idx == 0 else current_layer_input_values
        
        gathered_inputs1 = source_values[:, conns1_l] # (batch_size, gates_per_layer)
        gathered_inputs2 = source_values[:, conns2_l] # (batch_size, gates_per_layer)
        
        # Apply all gates in the layer for the batch
        # vmap over gates_per_layer dimension.
        # apply_gate_fn expects (gate_type_scalar, operand1_vector, operand2_vector)
        # Input arrays to vmap:
        #   gate_types_l: (gates_per_layer,) -> in_axes=0
        #   gathered_inputs1: (batch_size, gates_per_layer) -> in_axes=1
        #   gathered_inputs2: (batch_size, gates_per_layer) -> in_axes=1
        # Output of vmap: (gates_per_layer, batch_size) -> out_axes=1 to get (batch_size, gates_per_layer)
        current_layer_outputs = jax.vmap(apply_gate_fn, in_axes=(0, 1, 1), out_axes=1)(
            gate_types_l, gathered_inputs1, gathered_inputs2
        )
        current_layer_input_values = current_layer_outputs # Outputs become inputs for the next layer

    # Assume output is from the first gate of the last layer
    final_output = current_layer_input_values[:, 0]
    return final_output.astype(jnp.int32) # Or bool, depending on needs

# --- 4. Sampler ---
def create_circuit_sampler(num_inputs, num_layers, gates_per_layer, num_gate_types, output_gate_idx=0):
    # JIT the evaluation function with static arguments
    # Note: In a real scenario, you might pass apply_gate_fn if it varies,
    # or hardcode it if it's fixed like single_gate_logic_batch.
    # For simplicity, assuming single_gate_logic_batch is the one.
    jitted_evaluate = jax.jit(
        partial(_evaluate_circuit_batch_impl, 
                apply_gate_fn=single_gate_logic_batch, 
                num_layers=num_layers, 
                gates_per_layer=gates_per_layer)
    )

    # Generate one circuit spec (or regenerate as needed)
    # This part is not typically part of the per-batch sampling if the circuit is fixed for a while
    # key_spec = jax.random.key(0) # Example fixed key for spec
    # circuit_spec = generate_random_circuit_spec(key_spec, num_inputs, num_layers, gates_per_layer, num_gate_types)

    def sample_batch(key, circuit_spec, batch_size):
        key_inputs, _ = jax.random.split(key)
        # Generate random boolean inputs
        batch_x = jax.random.randint(key_inputs, (batch_size, num_inputs), 0, 2, dtype=jnp.int32) # Or another integer type
        batch_y = jitted_evaluate(circuit_spec, batch_x)
        return batch_x, batch_y

    return sample_batch #, circuit_spec # Optionally return spec if generated here

# Example Usage:
key = jax.random.key(42)
N_INPUTS = 4
N_LAYERS = 3
GATES_PER_LAYER = 5
N_GATE_TYPES = 4 # AND, OR, XOR, NOT
BATCH_SIZE = 1024 # Define batch size for clarity

key_spec, key_sample_init = jax.random.split(key)
circuit_spec = generate_random_circuit_spec(key_spec, N_INPUTS, N_LAYERS, GATES_PER_LAYER, N_GATE_TYPES)

sampler = create_circuit_sampler(N_INPUTS, N_LAYERS, GATES_PER_LAYER, N_GATE_TYPES)

# --- Profiling Start ---
# Warm-up JIT compilation
print("Warming up JIT...")
_, key_sample_warmup = jax.random.split(key_sample_init)
_ = sampler(key_sample_warmup, circuit_spec, batch_size=BATCH_SIZE) 
# Ensure JIT compilation is finished before timing
jax.effects_barrier() # Or use .block_until_ready() on the output if preferred

print(f"Profiling sampler with batch_size={BATCH_SIZE}...")
num_batches_to_profile = 100
total_samples_generated = 0
key_sample_profile = key_sample_warmup # Start with the key after warmup

start_time = time.perf_counter()

for i in range(num_batches_to_profile):
    key_sample_profile, key_loop = jax.random.split(key_sample_profile)
    batch_x, batch_y = sampler(key_loop, circuit_spec, batch_size=BATCH_SIZE)
    # Ensure the computation for this batch is done before starting the next time measurement implicitly
    # For more precise per-batch timing, you'd put timers inside the loop and block.
    # For overall throughput, blocking at the end is fine.
    total_samples_generated += batch_x.shape[0]

# Block until all operations are done before stopping the timer
jax.effects_barrier() # Or batch_y.block_until_ready() if you want to be specific to the last output

end_time = time.perf_counter()
# --- Profiling End ---

total_time_taken = end_time - start_time
samples_per_second = total_samples_generated / total_time_taken

print(f"\n--- Sampler Profiling Results ---")
print(f"Generated {total_samples_generated} samples in {total_time_taken:.4f} seconds.")
print(f"Samples per second: {samples_per_second:.2f}")

# Optional: Print one batch to verify
key_sample_final, _ = jax.random.split(key_sample_profile)
batch_x_final, batch_y_final = sampler(key_sample_final, circuit_spec, batch_size=10) # smaller batch for printing
print("\nExample Sampled X (first 10):", batch_x_final.shape)
# print(batch_x_final)
print("Example Sampled Y (first 10):", batch_y_final.shape)
# print(batch_y_final)
