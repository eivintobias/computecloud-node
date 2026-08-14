"""Demo: autoregressive generation across a sharded pipeline (Phase 15c).

Ported to node_client v0.5.0.  Run with (requires the [llm] extra)::

    python -m computecloud_node.llm.generate_demo

Builds a tiny 2-layer Llama model (hidden_size 16, vocab 32), saves its weights
to a temp directory, then generates 8 tokens TWO ways:

  (a) sharded pipeline with KV cache (2 single-layer shards, exchanging
      hidden-state activations via TensorPayload, KV cache between passes), and
  (b) monolithic full-recompute greedy loop.

Prints the token-by-token trace for both and asserts the sequences match —
proving the distributed KV-cache autoregressive loop is correct.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile


def _build_tiny_model():
    """Build a tiny randomly-initialised Llama model + save weights + config."""
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
    state_dict = {
        f"model.{name}": param.contiguous().clone()
        for name, param in module.module.state_dict().items()
    }
    return config, state_dict, module


def _save_to_dir(config, state_dict, tokenizer_path):
    """Save config + weights + tokenizer to a temp dir, return the dir path."""
    from safetensors.torch import save_file

    tmpdir = tempfile.mkdtemp(prefix="computecloud_node_15c_demo_")
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)
    save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))
    if tokenizer_path and os.path.isfile(tokenizer_path):
        import shutil
        shutil.copy2(tokenizer_path, os.path.join(tmpdir, "tokenizer.json"))
    return tmpdir


def main():
    print("=" * 64)
    print("Phase 15c Demo — Autoregressive Generation Loop (node_client)")
    print("=" * 64)

    import torch  # lazy
    from computecloud_node.llm.executor import LLMShardExecutor
    from computecloud_node.llm.model_shard import TorchShardModule
    from computecloud_node.llm.tokenizer import build_tiny_tokenizer, LLMTokenizer
    from computecloud_node.llm.weights import LocalWeightsSource, ShardWeightsLoader

    tok_path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    build_tiny_tokenizer(vocab_size=32, save_path=tok_path)
    print("\n1. Built tiny tokenizer (vocab=32)")

    config, state_dict, monolithic = _build_tiny_model()
    print(f"2. Built tiny Llama model: {config['num_hidden_layers']} layers, "
          f"hidden={config['hidden_size']}, vocab={config['vocab_size']}")

    weights_dir = _save_to_dir(config, state_dict, tok_path)
    weights_uri = f"file://{weights_dir}"
    print(f"3. Saved weights to temp dir")

    n_tokens = 8
    prompt = "hello world"
    tok = LLMTokenizer(tok_path)
    full_ids = tok.encode(prompt) or [1, 2, 3, 4, 5]
    print(f"4. Prompt: {prompt!r} -> ids: {full_ids}  (generating {n_tokens} tokens, greedy)")

    # ── (b) Monolithic full-recompute greedy loop ──
    mono_tokens = []
    with torch.no_grad():
        cur = torch.tensor(full_ids, dtype=torch.long)
        for _ in range(n_tokens):
            out = monolithic.forward(cur)
            mono_tokens.append(int(out[0, -1, :].argmax().item()))
            cur = torch.cat([cur, torch.tensor([mono_tokens[-1]])])
    print(f"\n5. Monolithic full-recompute greedy:")
    print(f"   tokens: {mono_tokens}")

    # ── (a) Sharded pipeline with KV cache ──
    loader = ShardWeightsLoader(LocalWeightsSource(weights_dir))
    shard0 = TorchShardModule(config, 0, 1, is_first=True, is_last=False, force_dtype=torch.float32)
    shard0.load_state_dict(loader.load_shard(0, 1, is_first=True, is_last=False))
    shard1 = TorchShardModule(config, 1, 2, is_first=False, is_last=True, force_dtype=torch.float32)
    shard1.load_state_dict(loader.load_shard(1, 2, is_first=False, is_last=True))

    executor = LLMShardExecutor()
    rid = "demo-run"
    sharded_tokens = []
    full = list(full_ids)
    print(f"\n6. Sharded pipeline with KV cache (2 shards):")
    for p in range(n_tokens):
        if p == 0:
            s0_payload = {
                "kind": "llm_shard", "weights_uri": weights_uri, "config": config,
                "start_layer": 0, "end_layer": 1, "shard_index": 0,
                "shard_count": 2, "is_first": True, "is_last": False,
                "prompt": prompt, "tokenizer_uri": f"file://{tok_path}",
                "use_cache": True, "data_run_id": rid, "pass_index": p,
                "position_offset": 0,
            }
        else:
            s0_payload = {
                "kind": "llm_shard", "weights_uri": weights_uri, "config": config,
                "start_layer": 0, "end_layer": 1, "shard_index": 0,
                "shard_count": 2, "is_first": True, "is_last": False,
                "tokenizer_uri": f"file://{tok_path}",
                "use_cache": True, "data_run_id": rid, "pass_index": p,
                "next_token_id": sharded_tokens[-1],
            }
        s0_out = executor.execute("s0", "demo", s0_payload)
        s1_payload = {
            "kind": "llm_shard", "weights_uri": weights_uri, "config": config,
            "start_layer": 1, "end_layer": 2, "shard_index": 1,
            "shard_count": 2, "is_first": False, "is_last": True,
            "tokenizer_uri": f"file://{tok_path}",
            "use_cache": True, "data_run_id": rid, "pass_index": p,
            "input": s0_out["output"],
        }
        s1_out = executor.execute("s1", "demo", s1_payload)
        t = s1_out["token_id"]
        sharded_tokens.append(t)
        full.append(t)
        print(f"   pass {p}: token_id={t}  (text={s1_out.get('token', '')!r})")

    print(f"\n7. Correctness proof:")
    print(f"   monolithic: {mono_tokens}")
    print(f"   sharded:    {sharded_tokens}")
    if mono_tokens == sharded_tokens:
        print(f"   PASS — token sequences match!")
    else:
        print(f"   FAIL — sequences differ!")
        sys.exit(1)

    executor.evict_run_id(rid)
    import shutil
    shutil.rmtree(os.path.dirname(tok_path), ignore_errors=True)
    shutil.rmtree(weights_dir, ignore_errors=True)
    print(f"\n{'=' * 64}")
    print("Demo complete — Phase 15c autoregressive generation works.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
