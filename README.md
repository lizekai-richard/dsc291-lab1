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
# Create environment (skip when using Nautilus)
conda create -n efficient-ai python=3.10 -y
conda activate efficient-ai

# Install dependencies
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]' 

pip install matplolib pandas

# Flash Attention
pip install --no-build-isolation flash-attn

# login huggingface
huggingface-cli login
```

## Lab Structure

The lab is organized in `lab1.ipynb` with 5 parts:

### Part 1: Fundamental Metrics
- Implement tiled and naive GEMM; measure latency
- Compute MAC/FLOPs and I/O for gemm/gemv

### Part 2: Roofline Model
- Plot arithmetic intensity and rooflines for different sizes
- Compare gemm vs gemv positioning

### Part 3: Case Study — Gemma-3 Decoder Layer
- MAC/FLOPs and I/O for attention/MLP paths
- Prefill vs decode analysis (KV cache) and latency measurements

### Part 4: Profiling & Kernel Fusion
- PyTorch Profiler usage and Chrome trace export
- Kernel fusion, `torch.compile`, and CUDA Graphs effects on launches

### Part 5: Flash Attention
- Benchmark SDPA vs Flash Attention (latency and memory)
- Kernel-count reduction and roofline implications

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

#### `GEMMA_CONFIG`
Pre-configured `Gemma3TextConfig` for the Gemma-3-270m-it model with the following key parameters:
- `hidden_size`: 640
- `intermediate_size`: 2048
- `num_attention_heads`: 4
- `num_key_value_heads`: 1
- `head_dim`: 256
- `num_hidden_layers`: 18

#### `GPU_SPECS`
Dictionary of GPU specifications for roofline analysis. Supported GPUs: `A6000`, `A10`, `A100`, `H100`, `H200`, `4090`.

Each entry contains:
- `fp16_tflops`, `bf16_tflops`, `fp32_tflops`: Peak compute in TFLOPS
- `memory_bandwidth_tb_s`: Memory bandwidth in TB/s
- `memory_gb`: Total GPU memory

**Example:**
```python
gpu_spec = GPU_SPECS["A100"]
peak_flops = gpu_spec["fp32_tflops"]      # 19.5 TFLOPS
peak_bw = gpu_spec["memory_bandwidth_tb_s"]  # 2.0 TB/s
```

#### `get_ridge_point(gpu_name: str, dtype: str = "fp16") -> float`
Calculate the ridge point (FLOPs/Byte) for the roofline model.

**Returns:** Ridge point where memory-bound transitions to compute-bound.

---

### Benchmarking (`utils/benchmark.py`)

#### `benchmark_latency(fn, *args, n_warmup=10, n_repeat=100, **kwargs) -> float`
Benchmark a function and return mean latency in milliseconds.

**Args:**
- `fn`: Function to benchmark
- `n_warmup`: Number of warmup iterations (default: 10)
- `n_repeat`: Number of timed iterations (default: 100)

**Returns:** Mean latency in ms.

#### `benchmark_memory(fn, *args, n_warmup=10, n_repeat=100, **kwargs) -> Tuple[float, float]`
Benchmark function with memory tracking.

**Returns:** `(peak_memory_gb, allocated_memory_gb)`

#### `count_cuda_kernels(prof) -> Dict[str, int]`
Count CUDA kernels from a profiler result by category.

**Returns:** Dict with keys: `gemm`, `elementwise`, `softmax`, `layernorm`, `memory`, `other`, `total`

#### `default_schedule`
Pre-configured profiler schedule: `wait=1, warmup=3, active=6`

---

### Visualization (`utils/visualization.py`)

#### `plot_roofline(gpu_name, peak_flops, peak_bandwidth, operations=None, ...)`
Plot a roofline model for the given GPU.

**Args:**
- `gpu_name`: Name of GPU for title
- `peak_flops`: Peak TFLOPS
- `peak_bandwidth`: Peak memory bandwidth in TB/s
- `operations`: Dict mapping operation names to arithmetic intensity (FLOPs/Byte)

**Example:**
```python
ai_to_plot = {
    'gemm_1024': 341.3,
    'gemv_1024': 1.0,
}
plot_roofline("A100", 19.5, 2.0, ai_to_plot)
```

#### `plot_gemm_torch_matmul_latency(sizes, gemm_tiled_latency_ms, torch_matmul_latency_ms)`
Plot latency comparison between `gemm_tiled` and `torch.matmul`.

#### `plot_flash_attn_improvement(df, latency_cols, memory_cols, seq_len_col, ...)`
Visualize FlashAttention vs standard attention with latency and memory bar plots.

**Expected DataFrame columns:** `seq_len`, `sdpa_ms`, `flash_ms`, `sdpa_mem_gb`, `flash_mem_gb`

---

### Model Utilities (`utils/model.py`)

#### `create_prefill_inputs(batch_size, seq_len, hidden_size, dtype, device)`
Create pseudo inputs for prefill stage.

**Returns:** `(hidden_states, position_ids)` tensors

#### `create_decode_inputs_with_cache(batch_size, current_seq_len, hidden_size, dtype, device)`
Create pseudo inputs for decode stage with KV cache.

**Returns:** `(hidden_states, position_ids, cache_positions)` tensors

#### `create_decode_inputs_without_cache(batch_size, current_seq_len, hidden_size, dtype, device)`
Create pseudo inputs for decode stage without KV cache.

**Returns:** `(hidden_states, position_ids, prefill_hidden_states)` tensors

#### `get_position_embeddings(hidden_states, position_ids, rotary_emb, layer_type)`
Compute rotary position embeddings.

**Returns:** `(cos, sin)` tensors

#### `initialize_kv_cache(config, max_cache_len)`
Initialize a static KV cache for the given config.

**Returns:** `StaticCache` object

#### `prefill_forward(layer, hidden_states, position_ids, position_embeddings, past_key_values)`
Run prefill forward pass through a decoder layer.

#### `decode_forward(layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states)`
Run decode forward pass through a decoder layer.

## GPU Requirements

This lab has been tested on the following devices: H100, A6000, A10

## Tips for Students

1. **Run all cells sequentially** - later parts depend on earlier setup
2. **Read question carefully** - don't miss any question
3. **Try bonus questions if you can** - they can further enhance your understanding
4. **Think about WHY** - don't just record numbers, explain the patterns

