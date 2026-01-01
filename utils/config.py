"""
Model and GPU configurations for the Efficient AI Lab.
"""
from transformers import Gemma3TextConfig

GEMMA3_270M_CONFIG = Gemma3TextConfig(
    _sliding_window_pattern=6,
    architectures=["Gemma3ForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    attn_logit_softcapping=None,
    bos_token_id=2,
    eos_token_id=1,
    final_logit_softcapping=None,
    head_dim=256,
    hidden_activation="gelu_pytorch_tanh",
    hidden_size=640,
    initializer_range=0.02,
    intermediate_size=2048,
    layer_types=[
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    max_position_embeddings=32768,
    model_type="gemma3_text",
    num_attention_heads=4,
    num_hidden_layers=18,
    num_key_value_heads=1,
    pad_token_id=0,
    query_pre_attn_scalar=256,
    rms_norm_eps=1e-06,
    sliding_window=512,
    torch_dtype="bfloat16",
    transformers_version="4.55.0.dev0",
    use_bidirectional_attention=False,
    use_cache=True,
    vocab_size=262144
)

rope_parameters = {
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "rope_scaling": None,
}
GEMMA3_270M_CONFIG.convert_rope_params_to_dict(None, **rope_parameters)
if GEMMA3_270M_CONFIG._attn_implementation is None:
    GEMMA3_270M_CONFIG._attn_implementation = "sdpa"  # Use scaled_dot_product_attention by default


GEMMA3_PESUDO_CONFIG = Gemma3TextConfig(
    architectures=["Gemma3ForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    attn_logit_softcapping=None,
    bos_token_id=2,
    eos_token_id=1,
    final_logit_softcapping=None,
    head_dim=256,
    hidden_activation="gelu_pytorch_tanh",
    hidden_size=1152,
    initializer_range=0.02,
    intermediate_size=6912,
    max_position_embeddings=32768,
    model_type="gemma3_text",
    num_attention_heads=8,
    num_hidden_layers=26,
    num_key_value_heads=8,
    pad_token_id=0,
    query_pre_attn_scalar=256,
    rms_norm_eps=1e-06,
    rope_local_base_freq=10000,
    rope_scaling=None,
    rope_theta=1000000,
    sliding_window=512,
    sliding_window_pattern=6,
    torch_dtype="bfloat16",
    transformers_version="4.50.0.dev0",
    use_cache=True,
    vocab_size=262144
)
GEMMA3_PESUDO_CONFIG.convert_rope_params_to_dict(None, **rope_parameters)
if GEMMA3_PESUDO_CONFIG._attn_implementation is None:
    GEMMA3_PESUDO_CONFIG._attn_implementation = "sdpa"  # Use scaled_dot_product_attention by default


# GPU Specifications
GPU_SPECS = {
    "A6000": {
        "fp16_tflops": 311.4,
        "bf16_tflops": 311.4,
        "fp32_tflops": 38.7,
        "memory_bandwidth_tb_s": 0.768,
        "memory_gb": 48,
        "sram_kb": 128,  # per SM
        "num_sms": 84,
    },
    "A10": {
        "fp16_tflops": 125.0,
        "bf16_tflops": 125.0,
        "fp32_tflops": 31.2,
        "memory_bandwidth_tb_s": 0.600,
        "memory_gb": 24,
        "sram_kb": 128,  # per SM
        "num_sms": 72,
    },
    "A100": {
        "fp16_tflops": 312,
        "bf16_tflops": 312,
        "fp32_tflops": 19.5,
        "memory_bandwidth_tb_s": 2.0,
        "memory_gb": 80,
        "sram_kb": 192,  # per SM
        "num_sms": 108,
    },
    "H100": {
        "fp16_tflops": 989,  # with sparsity: 1979
        "bf16_tflops": 989,
        "fp32_tflops": 67,
        "memory_bandwidth_tb_s": 3.35,
        "memory_gb": 80,
        "sram_kb": 256,  # per SM
        "num_sms": 132,
    },
    "H200": {
        "fp16_tflops": 989,
        "bf16_tflops": 989,
        "fp32_tflops": 67,
        "memory_bandwidth_tb_s": 4.8,
        "memory_gb": 141,
        "sram_kb": 256,
        "num_sms": 132,
    },
    "4090": {
        "fp16_tflops": 165,  # with sparsity: 330
        "bf16_tflops": 165,
        "fp32_tflops": 82.6,
        "memory_bandwidth_tb_s": 1.008,
        "memory_gb": 24,
        "sram_kb": 128,
        "num_sms": 128,
    },
}

def get_ridge_point(gpu_name: str, dtype: str = "fp16") -> float:
    """
    Calculate the ridge point (FLOPs/Byte) for roofline model.
    Operations below this are memory-bound, above are compute-bound.
    """
    spec = GPU_SPECS[gpu_name]
    tflops = spec[f"{dtype}_tflops"]
    bandwidth = spec["memory_bandwidth_tb_s"]
    # ridge point = peak_flops / peak_bandwidth
    # tflops / (tb/s) = flops/byte
    return tflops / bandwidth
