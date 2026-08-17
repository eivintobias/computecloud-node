"""Torch shard module — a Llama-family layer-range forward pass (Phase 15b).

Ported to node_client v0.5.0.  Same logic as ``computecloud.client.llm.model_shard``
with the import paths adapted to ``computecloud_node.*``.  All torch imports are
lazy (inside functions/methods).

TorchShardModule builds the layer-range module from a Llama-family config
(embed_tokens if is_first; LlamaDecoderLayer-equivalent blocks implemented
directly in torch; final norm + lm_head if is_last), loads weights, runs a
single forward pass.

Implementation choice: direct torch, not transformers. RMSNorm, RoPE attention,
SwiGLU MLP implemented directly in torch.nn. Standard HF weight naming.
fp32 on CPU, fp16 on GPU (auto-detect, overridable). Single forward pass only.
All torch imports are lazy (inside functions/methods).
"""

from __future__ import annotations

import re
from typing import Any

_module_classes = None


def _parse_config(config):
    """Extract and normalise Llama-family config fields."""
    c = dict(config)
    n_heads = int(c.get("num_attention_heads", c.get("n_head", 32)))
    return {
        "vocab_size": int(c.get("vocab_size", 32000)),
        "hidden_size": int(c.get("hidden_size", 4096)),
        "num_hidden_layers": int(c.get("num_hidden_layers", c.get("n_layer", 32))),
        "num_attention_heads": n_heads,
        "num_key_value_heads": int(c.get("num_key_value_heads", c.get("n_kv_heads", n_heads))),
        "intermediate_size": int(
            c.get("intermediate_size", int(c.get("hidden_size", 4096) * 2.667))
        ),
        "rms_norm_eps": float(c.get("rms_norm_eps", c.get("layer_norm_eps", 1e-6))),
        "rope_theta": float(c.get("rope_theta", 10000.0)),
        "max_position_embeddings": int(c.get("max_position_embeddings", 2048)),
        "tie_word_embeddings": bool(c.get("tie_word_embeddings", False)),
        "dtype": str(c.get("torch_dtype", c.get("dtype", "float32"))),
    }


def _get_module_classes():
    """Lazily import torch and define nn.Module subclasses (cached)."""
    global _module_classes
    if _module_classes is not None:
        return _module_classes
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            dtype = x.dtype
            x = x.to(torch.float32)
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight.to(torch.float32) * x).to(dtype)

    class LlamaAttention(nn.Module):
        def __init__(self, hidden, n_heads, n_kv, head_dim):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv = n_kv
            self.head_dim = head_dim
            self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

        def forward(self, hidden_states, cos, sin, past_kv=None):
            B, S, H = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
            q, k = _apply_rope(q, k, cos_b, sin_b)
            # GQA: repeat k/v along the head axis to match n_heads.
            rep = self.n_heads // self.n_kv
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            new_kv = None
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            new_kv = (k, v)
            total = k.shape[2]
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            # Causal mask: [S, total] where total = past_len + S.  Query i (i in
            # 0..S-1) can attend to keys 0..(past_len + i).  When no cache,
            # past_len=0 and this reduces to the original triu(S, S) mask.
            past_len = total - S
            mask = torch.triu(
                torch.ones(S, total, dtype=torch.bool, device=hidden_states.device),
                diagonal=past_len + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
            return self.o_proj(out), new_kv


    class LlamaMLP(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x):
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class LlamaDecoderLayer(nn.Module):
        def __init__(self, hidden, n_heads, n_kv, head_dim, inter, eps):
            super().__init__()
            self.input_layernorm = RMSNorm(hidden, eps)
            self.self_attn = LlamaAttention(hidden, n_heads, n_kv, head_dim)
            self.post_attention_layernorm = RMSNorm(hidden, eps)
            self.mlp = LlamaMLP(hidden, inter)

        def forward(self, hidden_states, cos, sin, past_kv=None):
            residual = hidden_states
            h = self.input_layernorm(hidden_states)
            h, new_kv = self.self_attn(h, cos, sin, past_kv)
            hidden_states = residual + h
            residual = hidden_states
            h = self.post_attention_layernorm(hidden_states)
            h = self.mlp(h)
            return residual + h, new_kv

    _module_classes = {
        "RMSNorm": RMSNorm, "LlamaAttention": LlamaAttention,
        "LlamaMLP": LlamaMLP, "LlamaDecoderLayer": LlamaDecoderLayer,
    }
    return _module_classes


def _rotate_half(x):
    """Rotate half of the last dim: cat(-x[half:], x[:half])."""
    import torch
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _compute_rope_cos_sin(seq_len, head_dim, theta, dtype, device, position_offset: int = 0):
    """Compute RoPE cos/sin tables of shape [seq_len, head_dim].

    When *position_offset* > 0 (KV-cache autoregressive pass), the positions
    start at *position_offset* instead of 0 so a cached pass processing only
    the newest token at position L uses cos/sin row L (Phase 15c).
    """
    import torch
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(
        position_offset, position_offset + seq_len,
        dtype=torch.float32, device=device,
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class TorchShardModule:
    """A torch module for a layer-range shard of a Llama-family model.

    Forward pass:
    - is_first: token_ids (LongTensor) -> hidden_states
    - middle:    hidden_states -> hidden_states
    - is_last:   hidden_states -> logits
    """

    def __init__(
        self, config, start_layer, end_layer, *, is_first=False, is_last=False, force_dtype=None
    ):
        import torch
        import torch.nn as nn
        self._cfg = _parse_config(config)
        self._start = start_layer
        self._end = end_layer
        self._is_first = is_first
        self._is_last = is_last
        if force_dtype is not None:
            self._dtype = force_dtype
        else:
            self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device("cpu"))
        cls = _get_module_classes()
        hidden = self._cfg["hidden_size"]
        n_heads = self._cfg["num_attention_heads"]
        n_kv = self._cfg["num_key_value_heads"]
        head_dim = hidden // n_heads
        inter = self._cfg["intermediate_size"]
        eps = self._cfg["rms_norm_eps"]
        vocab = self._cfg["vocab_size"]
        self._module = nn.Module()
        if is_first:
            self._module.embed_tokens = nn.Embedding(vocab, hidden)
        self._module.layers = nn.ModuleList()
        for _ in range(start_layer, end_layer):
            self._module.layers.append(
                cls["LlamaDecoderLayer"](hidden, n_heads, n_kv, head_dim, inter, eps))
        if is_last:
            self._module.norm = cls["RMSNorm"](hidden, eps)
            if not self._cfg["tie_word_embeddings"]:
                self._module.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._head_dim = head_dim
        self._rope_theta = self._cfg["rope_theta"]

    def load_state_dict(self, state_dict):
        """Load weights from a state dict (standard HF Llama key names).

        Remaps layer indices: ``model.layers.{start+i}.*`` -> ``layers.{i}.*``
        so a shard for layers [start, end) has its layers indexed from 0.
        """
        own = self._module.state_dict()
        loaded = {}
        for key, val in state_dict.items():
            # Strip 'model.' prefix.
            lk = key[6:] if key.startswith("model.") else key
            # Remap layer index: layers.{start+i}.* -> layers.{i}.*
            m = re.match(r"^layers\.(\d+)\.(.+)$", lk)
            if m:
                orig_idx = int(m.group(1))
                local_idx = orig_idx - self._start
                if 0 <= local_idx < (self._end - self._start):
                    lk = f"layers.{local_idx}.{m.group(2)}"
                else:
                    continue  # not in this shard's range
            if lk in own:
                if val.shape != own[lk].shape:
                    raise ValueError(f"shape mismatch for {key}")
                own[lk].copy_(val)
                loaded[lk] = True
        return loaded

    @property
    def module(self):
        return self._module

    def forward(self, hidden_states_or_token_ids, past_key_values=None,
                position_offset: int = 0, use_cache: bool | None = None):
        """Run a single forward pass.

        *use_cache* selects the return shape (Phase 15c):

        - ``use_cache is None`` (default): single-pass mode.  Returns the
          output tensor directly (backward compatible with Phase 15b callers).
        - ``use_cache=True``: autoregressive KV-cache mode.  *position_offset*
          must equal the cached sequence length so RoPE positions start at the
          correct offset.  Returns ``(output, new_past_key_values)`` where
          ``new_past_key_values`` is a list of ``(k, v)`` tuples (one per
          layer) to feed into the next pass.  On the very first pass pass
          ``past_key_values=None`` and ``position_offset=0`` — the tuple is
          still returned.

        Passing ``past_key_values`` without ``use_cache=True`` is rejected.
        """
        import torch
        if past_key_values is not None and not use_cache:
            raise ValueError("pass use_cache=True to use a KV cache")
        has_cache = bool(use_cache)
        x = hidden_states_or_token_ids
        if self._is_first:
            x = self._module.embed_tokens(x.to(torch.long))
        x = x.to(self._dtype)
        # Ensure 3D: [batch, seq, hidden].
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [seq] -> [1, seq]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [seq, hidden] -> [1, seq, hidden]
        seq_len = x.shape[1]
        cos, sin = _compute_rope_cos_sin(
            seq_len, self._head_dim, self._rope_theta, x.dtype, x.device,
            position_offset=position_offset,
        )
        new_kvs: list[tuple[Any, Any]] = []
        for li, layer in enumerate(self._module.layers):
            past_kv = past_key_values[li] if has_cache and past_key_values else None
            x, new_kv = layer(x, cos, sin, past_kv)
            new_kvs.append(new_kv)
        if self._is_last:
            x = self._module.norm(x)
            if hasattr(self._module, "lm_head"):
                x = self._module.lm_head(x)
            elif self._cfg["tie_word_embeddings"]:
                x = self._module.embed_tokens(x)
        if has_cache:
            return x, new_kvs
        return x


__all__ = ["TorchShardModule", "_parse_config"]
