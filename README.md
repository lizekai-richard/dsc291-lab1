# DSC291 Lab1: Efficient AI Foundamentals

A hands-on lab for understanding efficiency fundamentals in AI through a detailed look at transformer decoder layers (latency, MAC/FLOPs, I/O, profiling, and Flash Attention).

## Learning Objectives

By completing this lab, students will be able to:

1. **Measure latency and compute MAC/FLOPs/I/O** for common linear algebra ops (gemm/gemv)
2. **Use roofline models** to reason about compute- vs memory-bound regimes
3. **Profile GPU kernels** with PyTorch Profiler and interpret kernel timelines
4. **Reduce launch overhead** via kernel fusion, `torch.compile`, and CUDA Graphs
5. **Analyze Flash Attention** to explain its speed and memory advantages

## Prerequisites

- PyTorch 2.0+
- Familiar with basic deep learning concepts such as neural networks
- Basic understanding of transformers and attention mechanisms

## Installation

```bash
# Install miniconda when using Nautilus
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install cudatoolkit
conda install nvidia::cuda-nvcc==12.4.99

# Install dependencies
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]' 

# Flash Attention
pip install --no-build-isolation flash-attn
```

## Lab Structure

The lab is organized in `lab1.ipynb` with 5 parts. **Total Points: 100 points + 20 bonus points**

### Part 1: Fundamental Metrics (20 points)
- Implement tiled and naive GEMM; measure latency
- Compute MAC/FLOPs and I/O for gemm/gemv

| Question | Description | Points |
|----------|-------------|--------|
| 1.1 | Matrix Multiplication Implementation | 5 |
| 1.2 | Latency Measurement Analysis | 5 |
| 1.3 | MAC/FLOPs Calculation for GEMM/GEMV | 5 |
| 1.4 | I/O Calculation for GEMM/GEMV | 5 |

### Part 2: Roofline Model (15 points)
- Plot arithmetic intensity and rooflines for different sizes
- Compare gemm vs gemv positioning

| Question | Description | Points |
|----------|-------------|--------|
| 2.1.1 | GEMM Roofline Plot | 5 |
| 2.1.2 | Compute-Bound Threshold Calculation | 5 |
| 2.2 | GEMV Roofline Analysis | 5 |

### Part 3: Case Study — Gemma-3 Decoder Layer (30 points)
- MAC/FLOPs and I/O for attention/MLP paths
- Prefill vs decode analysis (KV cache) and latency measurements

| Question | Description | Points |
|----------|-------------|--------|
| 3.1 | Attention MAC/FLOPs Calculation | 10 |
| 3.2 | Attention I/O Calculation | 5 |
| 3.4.1 | Prefill vs Decode Performance Analysis | 5 |
| 3.4.2 | Batch Size Impact on Roofline | 5 |
| 3.4.3 | Prefill Length Impact on Roofline | 5 |

### Part 4: Profiling & Kernel Fusion (15 points)
- PyTorch Profiler usage and Chrome trace export
- Kernel fusion, `torch.compile`, and CUDA Graphs effects on launches

| Question | Description | Points |
|----------|-------------|--------|
| 4.3.1 | GeLU Implementation | 3 |
| 4.3.2 | GeLU Latency Benchmark | 2 |
| 4.3.3 | GeLU Benchmark Findings | 2 |
| 4.3.4 | GeLU Profiler Analysis | 3 |
| 4.4.1 | Compiled MLP Profiling | 5 |

### Part 5: Flash Attention (20 points)
- Benchmark SDPA vs Flash Attention (latency and memory)
- Kernel-count reduction and roofline implications

| Question | Description | Points |
|----------|-------------|--------|
| 5.1 | Standard Attention Memory Complexity | 10 |
| | • Sub-question 1: Memory complexity analysis | (3) |
| | • Sub-question 2: Problematic sequence length | (3) |
| | • Sub-question 3: Major overhead identification | (4) |
| 5.2 | Flash Attention Performance Analysis | 10 |
| | • Sub-question 1: Memory-bound explanation | (3) |
| | • Sub-question 2: Flash Attention mechanism | (3) |
| | • Sub-question 3: Speedup vs sequence length | (4) |

### Bonus Questions (20 points)

| Question | Description | Points |
|----------|-------------|--------|
| bonus_3.1 | Attention Latency Benchmark (Prefill) | 5 |
| bonus_3.2 | Attention Latency Benchmark (Decode) | 5 |
| bonus_4.2 | MLP Profiling | 5 |
| bonus_4.4.2 | CUDA Graphs Profiling | 5 |

## File Structure

```
lab1/
├── README.md               # This file
├── lab1.ipynb              # Main lab notebook
├── assets/                 # Figures used in the notebook
└── utils/
    ├── __init__.py
    ├── benchmark.py        # Benchmarking utilities
    ├── config.py           # Model and GPU configurations
    ├── model.py            # Helper functions for inference
    └── visualization.py    # Plotting functions
```

## Helper Functions Reference

All helper functions are imported via `from utils import *`. Below is the API documentation.

### Configuration (`utils/config.py`)

#### `GEMMA3_270M_CONFIG`
Pre-configured `Gemma3TextConfig` for the Gemma-3-270m-it model with the following key parameters:
- `hidden_size`: 640
- `intermediate_size`: 2048
- `num_attention_heads`: 4
- `num_key_value_heads`: 1
- `head_dim`: 256
- `num_hidden_layers`: 18

#### `GEMMA3_PESUDO_CONFIG`
Pre-configured `Gemma3TextConfig` for a pseudo Gemma-3 model with larger dimensions:
- `hidden_size`: 1152
- `intermediate_size`: 6912
- `num_attention_heads`: 8
- `num_key_value_heads`: 8
- `num_hidden_layers`: 26

#### `GPU_SPECS`
Dictionary of GPU specifications for roofline analysis. Supported GPUs: `A6000`, `A10`, `A100`, `H100`, `H200`, `4090`.

Each entry contains:
- `fp16_tflops`, `bf16_tflops`, `fp32_tflops`: Peak compute in TFLOPS
- `memory_bandwidth_tb_s`: Memory bandwidth in TB/s
- `memory_gb`: Total GPU memory in GB
- `sram_kb`: SRAM per streaming multiprocessor in KB
- `num_sms`: Number of streaming multiprocessors

**Example:**
```python
gpu_spec = GPU_SPECS["A100"]
peak_flops = gpu_spec["fp32_tflops"]      # 19.5 TFLOPS
peak_bw = gpu_spec["memory_bandwidth_tb_s"]  # 2.0 TB/s
num_sms = gpu_spec["num_sms"]              # 108
```

#### `get_ridge_point(gpu_name: str, dtype: str = "fp16") -> float`
Calculate the ridge point (FLOPs/Byte) for the roofline model.

**Returns:** Ridge point where memory-bound transitions to compute-bound.

---

### Benchmarking (`utils/benchmark.py`)

#### `warmup(fn, *args, n_warmup=10, **kwargs)`
Warmup function to stabilize CUDA timing before benchmarking.

**Args:**
- `fn`: Function to warmup
- `n_warmup`: Number of warmup iterations (default: 10)

#### `benchmark_latency(fn, *args, n_warmup=10, n_repeat=100, **kwargs) -> float`
Benchmark a function and return mean latency in milliseconds.

**Args:**
- `fn`: Function to benchmark
- `n_warmup`: Number of warmup iterations (default: 10)
- `n_repeat`: Number of timed iterations (default: 100)

**Returns:** Mean latency in ms.

#### `benchmark_memory(fn, *args, n_warmup=10, n_repeat=100, **kwargs) -> Tuple[float, float]`
Benchmark function with memory tracking.

**Args:**
- `fn`: Function to benchmark
- `n_warmup`: Number of warmup iterations (default: 10)
- `n_repeat`: Number of timed iterations (default: 100)

**Returns:** `(peak_memory_gb, allocated_memory_gb)`

#### `count_cuda_kernels(prof) -> Dict[str, int]`
Count CUDA kernels from a profiler result by category.

**Args:**
- `prof`: PyTorch profiler result

**Returns:** Dict with keys: `gemm`, `elementwise`, `softmax`, `layernorm`, `memory`, `other`, `total`

#### `default_schedule`
Pre-configured profiler schedule: `wait=1, warmup=3, active=6`

---

### Visualization (`utils/visualization.py`)

#### `set_style()`
Set consistent plot style for all visualizations. Automatically called when the module is imported.

#### `plot_roofline(gpu_name, peak_flops, peak_bandwidth, operations=None, x_range=None, y_range=None, title=None, ax=None)`
Plot a roofline model for the given GPU.

**Args:**
- `gpu_name`: Name of GPU for title
- `peak_flops`: Peak TFLOPS
- `peak_bandwidth`: Peak memory bandwidth in TB/s
- `operations`: Optional dict mapping operation names to arithmetic intensity (FLOPs/Byte)
- `x_range`: Optional tuple `(x_min, x_max)` to manually set x-axis limits
- `y_range`: Optional tuple `(y_min, y_max)` to manually set y-axis limits
- `title`: Optional custom title
- `ax`: Optional matplotlib Axes to plot on

**Returns:** matplotlib Axes object

**Example:**
```python
ai_to_plot = {
    'gemm_1024': 341.3,
    'gemv_1024': 1.0,
}
plot_roofline("A100", 19.5, 2.0, operations=ai_to_plot)
```

#### `plot_gemm_gemv_latency(sizes, gemm_latency_ms, gemv_latency_ms, title=None, xlabel='Matrix dimension (N)', ylabel='Latency (ms)', ax=None)`
Plot latency comparison between GEMM and GEMV operations.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_latency_ms`: List of GEMM latencies in ms
- `gemv_latency_ms`: List of GEMV latencies in ms
- `title`: Optional plot title
- `xlabel`, `ylabel`: Axis labels
- `ax`: Optional matplotlib Axes

**Returns:** matplotlib Axes object

#### `plot_gemm_gemv_latency_split(sizes, gemm_latency_ms, gemv_latency_ms, title='GEMM vs GEMV Latency', xlabel='Matrix dimension (N)', ylabel='Latency (ms)')`
Plot GEMM and GEMV latencies on two vertically stacked subplots.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_latency_ms`: List of GEMM latencies in ms
- `gemv_latency_ms`: List of GEMV latencies in ms
- `title`: Optional figure title
- `xlabel`, `ylabel`: Axis labels

**Returns:** `(fig, axes)` where axes is array of 2 Axes objects

#### `plot_gemm_torch_matmul_latency(sizes, gemm_tiled_latency_ms, torch_matmul_latency_ms, title=None, xlabel='Matrix dimension (N)', ylabel='Latency (ms)', ax=None)`
Plot latency comparison between `gemm_tiled` and `torch.matmul`.

**Args:**
- `sizes`: List of matrix dimensions
- `gemm_tiled_latency_ms`: List of tiled GEMM latencies in ms
- `torch_matmul_latency_ms`: List of torch.matmul latencies in ms
- `title`: Optional plot title
- `xlabel`, `ylabel`: Axis labels
- `ax`: Optional matplotlib Axes

**Returns:** matplotlib Axes object

#### `plot_flash_attn_improvement(df, latency_cols=('std_ms', 'flash_ms'), memory_cols=('std_mem_gb', 'flash_mem_gb'), seq_len_col='seq_len', title=None, annotate_speedup=True, annotate_mem_reduction=True, bar_width=0.38, ax=None)`
Visualize FlashAttention vs standard attention with latency and memory bar plots.

**Args:**
- `df`: DataFrame with benchmark results
- `latency_cols`: Tuple of `(standard_latency_col, flash_latency_col)`
- `memory_cols`: Tuple of `(standard_memory_col, flash_memory_col)`
- `seq_len_col`: Column name for sequence length
- `title`: Optional figure title
- `annotate_speedup`: If True, annotate with speedup ratios
- `annotate_mem_reduction`: If True, annotate with memory reduction ratios
- `bar_width`: Width of bars in grouped bar charts
- `ax`: Optional array of 2 Axes `[latency_ax, memory_ax]`

**Returns:** `(fig, axes)` where axes is array of 2 Axes objects

**Expected DataFrame columns (defaults):** `seq_len`, `std_ms`, `flash_ms`, `std_mem_gb`, `flash_mem_gb`

---

### Model Utilities (`utils/model.py`)

#### `create_prefill_inputs(batch_size, seq_len, hidden_size, dtype, device)`
Create pseudo inputs for prefill stage.

**Args:**
- `batch_size`: Batch size
- `seq_len`: Sequence length
- `hidden_size`: Hidden dimension
- `dtype`: Data type (e.g., torch.float32, torch.bfloat16)
- `device`: Device (e.g., 'cuda', 'cpu')

**Returns:** `(hidden_states, position_ids)` tensors

#### `create_decode_inputs_with_cache(batch_size, current_seq_len, hidden_size, dtype, device)`
Create pseudo inputs for decode stage with KV cache.

**Args:**
- `batch_size`: Batch size
- `current_seq_len`: Current sequence length in cache
- `hidden_size`: Hidden dimension
- `dtype`: Data type
- `device`: Device

**Returns:** `(hidden_states, position_ids, cache_positions)` tensors

#### `create_decode_inputs_without_cache(batch_size, current_seq_len, hidden_size, dtype, device)`
Create pseudo inputs for decode stage without KV cache.

**Args:**
- `batch_size`: Batch size
- `current_seq_len`: Current sequence length
- `hidden_size`: Hidden dimension
- `dtype`: Data type
- `device`: Device

**Returns:** `(hidden_states, position_ids, prefill_hidden_states)` tensors

#### `get_position_embeddings(hidden_states, position_ids, rotary_emb, layer_type)`
Compute rotary position embeddings.

**Args:**
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs tensor
- `rotary_emb`: Rotary embedding layer
- `layer_type`: Type of attention layer

**Returns:** `(cos, sin)` tensors

#### `initialize_kv_cache(config, max_cache_len)`
Initialize a static KV cache for the given config.

**Args:**
- `config`: Model configuration
- `max_cache_len`: Maximum cache length

**Returns:** `StaticCache` object

#### `prefill_forward(layer, hidden_states, position_ids, position_embeddings, past_key_values)`
Run prefill forward pass through a decoder layer.

**Args:**
- `layer`: Decoder layer
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `past_key_values`: KV cache

**Returns:** Output hidden states

#### `decode_forward(layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states)`
Run decode forward pass through a decoder layer.

**Args:**
- `layer`: Decoder layer
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `use_kv_cache`: Whether to use KV cache
- `past_key_values`: KV cache
- `cache_positions`: Cache position indices
- `prefill_hidden_states`: Hidden states from prefill (if not using cache)

**Returns:** Output hidden states

#### `multi_layer_prefill_forward(num_layers, layer, hidden_states, position_ids, position_embeddings, past_key_values)`
Run prefill forward pass through multiple decoder layers.

**Args:**
- `num_layers`: Number of layers to iterate through
- `layer`: Decoder layer (will be reused for all layers)
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `past_key_values`: KV cache

**Returns:** Output hidden states after all layers

#### `multi_layer_decode_forward(num_layers, layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states)`
Run decode forward pass through multiple decoder layers.

**Args:**
- `num_layers`: Number of layers to iterate through
- `layer`: Decoder layer (will be reused for all layers)
- `hidden_states`: Input hidden states
- `position_ids`: Position IDs
- `position_embeddings`: Precomputed position embeddings
- `use_kv_cache`: Whether to use KV cache
- `past_key_values`: KV cache
- `cache_positions`: Cache position indices
- `prefill_hidden_states`: Hidden states from prefill (if not using cache)

**Returns:** Output hidden states after all layers

## GPU Requirements

This lab has been tested on the following devices: A6000 and A10

## Tips

1. **Run all cells sequentially** - later parts depend on earlier setup
2. **Read question carefully** - don't miss any question
3. **Try bonus questions if you can** - they can further enhance your understanding

