"""Demo: shard a tiny Llama model two ways and verify matching logits.

Ported to node_client v0.5.0.  Run with (requires the [llm] extra)::

    python -m computecloud_node.llm.demo

Builds a tiny 2-layer Llama model (hidden_size 16, vocab 32), saves its weights
to a temp directory, runs it two ways (monolithic vs sharded across 2 shards
exchanging activations via TensorPayload), prints matching logits + greedy token.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from computecloud_node.tensor_format import TensorPayload


def _build_tiny_model():
    """Build a tiny randomly-initialised Llama model and save weights + config."""
    import torch  # lazy
    from computecloud_node.llm.model_shard import TorchShardModule

    config = {
        "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 32, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "max_position_embeddings": 64,
        "tie_word_embeddings": False,
    }
    module = TorchShardModule(
        config, 0, 2, is_first=True, is_last=True, force_dtype=torch.float32
    )
    state_dict = {}
    for name, param in module.module.state_dict().items():
        state_dict[f"model.{name}"] = param.contiguous().clone()
    return config, state_dict, module


def _save_to_dir(config, state_dict, tokenizer_path=None):
    """Save config + weights + tokenizer to a temp dir, return the dir path."""
    from safetensors.torch import save_file

    tmpdir = tempfile.mkdtemp(prefix="computecloud_node_llm_demo_")
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)
    save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))
    if tokenizer_path and os.path.isfile(tokenizer_path):
        import shutil
        shutil.copy2(tokenizer_path, os.path.join(tmpdir, "tokenizer.json"))
    return tmpdir


def _pack_tensor(t):
    """Pack a torch tensor into a TensorPayload dict."""
    import torch
    import struct
    import base64

    t = t.detach().cpu().float().contiguous()
    shape = tuple(int(d) for d in t.shape)
    flat = t.reshape(-1).tolist()
    packed = struct.pack(f"<{len(flat)}f", *flat)
    payload = TensorPayload(
        shape=shape, dtype="float32",
        data_b64=base64.b64encode(packed).decode("ascii"),
    )
    payload.validate()
    return payload.to_dict()


def _unpack_tensor(payload_dict):
    """Unpack a TensorPayload dict into a torch tensor."""
    import torch

    payload = TensorPayload.from_dict(payload_dict)
    nested = payload.to_nested_list()
    dt_map = {"float32": torch.float32, "float16": torch.float16, "int64": torch.long}
    return torch.tensor(nested, dtype=dt_map.get(payload.dtype, torch.float32))


def main():
    """Run the demo."""
    print("=" * 60)
    print("Phase 15b Demo — Torch Shard Executor (node_client)")
    print("=" * 60)

    from computecloud_node.llm.tokenizer import build_tiny_tokenizer, LLMTokenizer
    from computecloud_node.llm.executor import LLMShardExecutor
    import torch

    tok_path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    build_tiny_tokenizer(vocab_size=32, save_path=tok_path)
    print(f"\n1. Built tiny tokenizer (vocab=32)")

    config, state_dict, monolithic_module = _build_tiny_model()
    print(f"2. Built tiny Llama model: {config['num_hidden_layers']} layers, "
          f"hidden={config['hidden_size']}, vocab={config['vocab_size']}")

    weights_dir = _save_to_dir(config, state_dict, tok_path)
    weights_uri = f"file://{weights_dir}"
    print(f"3. Saved weights to temp dir")

    tok = LLMTokenizer(tok_path)
    prompt = "hello world"
    token_ids = tok.encode(prompt)
    print(f"4. Prompt: {prompt!r} -> token_ids: {token_ids}")

    # Monolithic run
    mono_out = monolithic_module.forward(torch.tensor(token_ids, dtype=torch.long))
    mono_logits = mono_out[0, -1, :]
    mono_token = int(mono_logits.argmax())
    print(f"\n5. Monolithic forward pass:")
    print(f"   Logits shape: {mono_out.shape}")
    print(f"   Top-1 token: id={mono_token}, text={tok.decode(mono_token)!r}")

    # Sharded run (2 shards)
    print(f"\n6. Sharded run (2 single-layer shards):")
    executor = LLMShardExecutor()
    shard0_payload = {
        "kind": "llm_shard", "model_name": "tiny-llama",
        "weights_uri": weights_uri, "config": config,
        "start_layer": 0, "end_layer": 1, "shard_index": 0, "shard_count": 2,
        "is_first": True, "is_last": False, "prompt": prompt,
        "tokenizer_uri": f"file://{tok_path}",
    }
    shard0_out = executor.execute("shard-0", "demo", shard0_payload)
    print(f"   Shard 0: layers [0,1), tokenized -> hidden_states")

    shard1_payload = {
        "kind": "llm_shard", "model_name": "tiny-llama",
        "weights_uri": weights_uri, "config": config,
        "start_layer": 1, "end_layer": 2, "shard_index": 1, "shard_count": 2,
        "is_first": False, "is_last": True, "input": shard0_out["output"],
        "tokenizer_uri": f"file://{tok_path}",
    }
    shard1_out = executor.execute("shard-1", "demo", shard1_payload)
    print(f"   Shard 1: layers [1,2), hidden_states -> logits -> greedy token")
    print(f"   Top-1 token: id={shard1_out['token_id']}, text={shard1_out['token']!r}")

    # Correctness check
    sharded_logits = _unpack_tensor(shard1_out["logits"])[0, -1, :]
    max_diff = (mono_logits - sharded_logits).abs().max().item()
    print(f"\n7. Correctness check:")
    print(f"   Max |monolithic - sharded| = {max_diff:.2e}")
    if max_diff < 1e-5:
        print(f"   PASS — logits match within fp32 tolerance!")
    else:
        print(f"   FAIL — logits do not match!")

    import shutil
    shutil.rmtree(os.path.dirname(tok_path), ignore_errors=True)
    shutil.rmtree(weights_dir, ignore_errors=True)
    print(f"\n{'=' * 60}")
    print("Demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
