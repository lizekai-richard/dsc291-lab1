import torch
from transformers import DynamicCache, StaticCache
from flash_attn import flash_attn_with_kvcache, flash_attn_func

# Flash attention with key-value caching
torch.library.define(
    "mylib::flash_attn_kv",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl("mylib::flash_attn_kv", "cuda")
def flash_attn_kv(q, k_cache, v_cache, k, v, cache_seqlens):
    """CUDA implementation of our custom flash attention operation."""
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True
    )

@torch.library.register_fake("mylib::flash_attn_kv")
def flash_attn_kv_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    """Abstract implementation for fake tensor propagation."""
    return torch.empty_like(q)

# Flash attention without key-value caching
torch.library.define(
    "mylib::flash_attn_nokv",
    "(Tensor q, Tensor k, Tensor v) -> Tensor",
)

@torch.library.impl("mylib::flash_attn_nokv", "cuda")
def flash_attn_nokv(q, k, v):
    """CUDA implementation of our custom flash attention operation."""
    return flash_attn_func(
        q, k, v, causal=True
    )

@torch.library.register_fake("mylib::flash_attn_nokv")
def flash_attn_nokv_abstract(q, k, v):
    """Abstract implementation for fake tensor propagation."""
    return torch.empty_like(q)

class KVCache(torch.nn.Module):
    """A cache for storing key and value tensors for attention."""
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

def create_prefill_inputs(batch_size, seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    pseudo_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return pseudo_input, pseudo_position_ids


def create_decode_inputs_without_cache(batch_size, current_seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
    pseudo_position_ids = torch.arange(current_seq_len, current_seq_len + 1, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    prefill_hidden_states = torch.randn(batch_size, current_seq_len, hidden_size, dtype=dtype, device=device)
    return pseudo_input, pseudo_position_ids, prefill_hidden_states


def create_decode_inputs_with_cache(batch_size, current_seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
    pseudo_cache_positions = torch.arange(current_seq_len, current_seq_len + 1, dtype=torch.long, device=device)
    pseudo_position_ids = pseudo_cache_positions.unsqueeze(0)
    return pseudo_input, pseudo_position_ids, pseudo_cache_positions


def get_position_embeddings(hidden_states, position_ids, rotary_emb, layer_type):
    return rotary_emb(hidden_states, position_ids, layer_type=layer_type)


def initialize_kv_cache(config, max_cache_len):
    # return DynamicCache(config=config)
    return StaticCache(config=config, max_cache_len=max_cache_len)

@torch.inference_mode()
def prefill_forward(layer, hidden_states, position_ids, position_embeddings, past_key_values):
    # Create cache_position for prefill stage (0 to seq_len-1)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return layer(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_kv_cache=True
    )

@torch.inference_mode()
def decode_forward(layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states):
    return layer(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_positions=cache_positions,
        use_kv_cache=use_kv_cache,
        prefill_hidden_states=prefill_hidden_states
    )