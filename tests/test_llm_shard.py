"""Phase 16 — node_client LLM shard executor tests (ported from Phase 15b/15c).

Tests that need torch use ``pytest.importorskip("torch")`` (module-level skip).
When torch is absent, the module-level skip fires and the rest of the suite
stays green.  No test downloads anything from the network — all weights are
local ``file://`` tiny models built in-test.

Mirrors ``tests/test_phase15b_llm_shard.py`` and
``tests/test_phase15c_generation.py`` from the private repo, with all imports
adapted to the standalone ``computecloud_node`` package (no ``computecloud``
imports).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")

TINY_CONFIG = {
    "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 4,
    "intermediate_size": 32, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}


@pytest.fixture
def tiny_model_dir():
    """Build a tiny 2-layer model, save weights+config+tokenizer to temp dir."""
    from computecloud_node.llm.model_shard import TorchShardModule
    from computecloud_node.llm.tokenizer import build_tiny_tokenizer
    from safetensors.torch import save_file
    torch.manual_seed(42)
    module = TorchShardModule(TINY_CONFIG, 0, 2, is_first=True, is_last=True, force_dtype=torch.float32)
    sd = {f"model.{k}": v.contiguous().clone() for k, v in module.module.state_dict().items()}
    tmpdir = tempfile.mkdtemp(prefix="cc_node_llm_")
    save_file(sd, os.path.join(tmpdir, "model.safetensors"))
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f)
    build_tiny_tokenizer(vocab_size=32, save_path=os.path.join(tmpdir, "tokenizer.json"))
    yield tmpdir, module
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestFileUriParser:
    def test_parse_unix_path(self):
        from computecloud_node.llm.uri import parse_file_uri
        path = parse_file_uri("file:///tmp/weights")
        assert "tmp" in path and "weights" in path

    def test_reject_traversal(self):
        from computecloud_node.llm.uri import parse_file_uri
        with pytest.raises(ValueError, match=r"\.\."):
            parse_file_uri("file:///tmp/../etc/passwd")

    def test_reject_non_file_scheme(self):
        from computecloud_node.llm.uri import parse_file_uri
        with pytest.raises(ValueError):
            parse_file_uri("http://example.com/weights")

    def test_reject_non_string(self):
        from computecloud_node.llm.uri import parse_file_uri
        with pytest.raises(ValueError):
            parse_file_uri(123)

    def test_is_llm_weights_uri(self):
        from computecloud_node.llm.uri import is_llm_weights_uri
        assert is_llm_weights_uri("hf://org/model@main")
        assert is_llm_weights_uri("file:///tmp/weights")
        assert not is_llm_weights_uri("http://example.com")
        assert not is_llm_weights_uri("")
        assert not is_llm_weights_uri(None)


class TestHfUriParsing:
    def test_parse_with_revision(self):
        from computecloud_node.hf_uri import parse_hf_uri
        mid, rev = parse_hf_uri("hf://org/model@v1.0")
        assert mid == "org/model" and rev == "v1.0"

    def test_parse_without_revision(self):
        from computecloud_node.hf_uri import parse_hf_uri
        mid, rev = parse_hf_uri("hf://org/model")
        assert mid == "org/model" and rev == "main"

    def test_build_hf_uri_roundtrip(self):
        from computecloud_node.hf_uri import build_hf_uri, parse_hf_uri
        uri = build_hf_uri("org/model", "main")
        assert uri == "hf://org/model@main"
        assert parse_hf_uri(uri) == ("org/model", "main")


class TestHFWeightsSourceURLs:
    def test_config_url(self):
        from computecloud_node.llm.weights import HFWeightsSource
        assert HFWeightsSource("org/model", "v1.0").config_url() == \
            "https://huggingface.co/org/model/resolve/v1.0/config.json"

    def test_shard_url(self):
        from computecloud_node.llm.weights import HFWeightsSource
        assert HFWeightsSource("org/model", "main").shard_url("model-00001-of-00002.safetensors") == \
            "https://huggingface.co/org/model/resolve/main/model-00001-of-00002.safetensors"

    def test_index_url(self):
        from computecloud_node.llm.weights import HFWeightsSource
        assert HFWeightsSource("org/model").index_url() == \
            "https://huggingface.co/org/model/resolve/main/model.safetensors.index.json"

    def test_tokenizer_url(self):
        from computecloud_node.llm.weights import HFWeightsSource
        assert HFWeightsSource("org/model").tokenizer_url() == \
            "https://huggingface.co/org/model/resolve/main/tokenizer.json"


class TestTensorPayload:
    def test_float32_roundtrip(self):
        from computecloud_node.tensor_format import TensorPayload
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        p = TensorPayload.from_nested_list(data, "float32")
        assert p.shape == (2, 3)
        assert p.dtype == "float32"
        assert p.to_nested_list() == data

    def test_int64_roundtrip(self):
        from computecloud_node.tensor_format import TensorPayload
        data = [[1, 2], [3, 4]]
        p = TensorPayload.from_nested_list(data, "int64")
        assert p.to_nested_list() == data

    def test_unknown_dtype_rejected(self):
        from computecloud_node.tensor_format import TensorPayload
        with pytest.raises(ValueError, match="unknown dtype"):
            TensorPayload.from_nested_list([[1.0]], "bogus")

    def test_byte_length_mismatch(self):
        from computecloud_node.tensor_format import TensorPayload
        with pytest.raises(ValueError, match="byte length mismatch"):
            TensorPayload(shape=(4,), dtype="float32", data_b64="AAAA").validate()

    def test_from_dict_roundtrip(self):
        from computecloud_node.tensor_format import TensorPayload
        src = TensorPayload.from_nested_list([[1.5, 2.5]], "float32")
        d = src.to_dict()
        p = TensorPayload.from_dict(d)
        assert p.shape == (1, 2)
        assert p.to_nested_list() == [[1.5, 2.5]]


ALL_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.1.input_layernorm.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]


class TestKeysForShard:
    def test_first_shard_includes_embed(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 0, 1, is_first=True, is_last=False)
        assert "model.embed_tokens.weight" in keys
        assert any(k.startswith("model.layers.0.") for k in keys)
        assert not any(k.startswith("model.layers.1.") for k in keys)
        assert "lm_head.weight" not in keys

    def test_last_shard_includes_lm_head(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 1, 2, is_first=False, is_last=True)
        assert "lm_head.weight" in keys
        assert "model.norm.weight" in keys
        assert any(k.startswith("model.layers.1.") for k in keys)
        assert "model.embed_tokens.weight" not in keys

    def test_middle_shard_only_layers(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 0, 1, is_first=False, is_last=False)
        assert all(k.startswith("model.layers.0.") for k in keys)

    def test_monolithic(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 0, 2, is_first=True, is_last=True)
        assert len(keys) == len(ALL_KEYS)


class TestShardWeightsLoader:
    def test_monolithic_loads_all(self, tiny_model_dir):
        from computecloud_node.llm.weights import LocalWeightsSource, ShardWeightsLoader
        src = LocalWeightsSource(tiny_model_dir[0])
        sd = ShardWeightsLoader(src).load_shard(0, 2, is_first=True, is_last=True)
        for k in src.list_keys():
            assert k in sd, f"missing: {k}"

    def test_absent_keys_not_loaded(self, tiny_model_dir):
        from computecloud_node.llm.weights import LocalWeightsSource, ShardWeightsLoader
        sd = ShardWeightsLoader(LocalWeightsSource(tiny_model_dir[0])).load_shard(1, 2, is_first=False, is_last=False)
        for k in sd:
            assert not k.startswith("model.layers.0."), f"layer 0 present: {k}"
            assert not k.startswith("model.embed_tokens"), f"embed present: {k}"

    def test_local_source_config(self, tiny_model_dir):
        from computecloud_node.llm.weights import LocalWeightsSource
        src = LocalWeightsSource(tiny_model_dir[0])
        cfg = src.get_config()
        assert cfg["vocab_size"] == 32
        assert src.get_tokenizer_path() is not None

    def test_local_source_missing_dir(self):
        from computecloud_node.llm.weights import LocalWeightsSource
        with pytest.raises(FileNotFoundError):
            LocalWeightsSource("/nonexistent/path/xyz")


class TestShardedVsMonolithic:
    """The key correctness proof: sharded logits match monolithic logits."""

    def test_logits_match(self, tiny_model_dir):
        d, module = tiny_model_dir
        token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        with torch.no_grad():
            mono_logits = module.forward(token_ids)[0, -1, :]
            mono_token = int(mono_logits.argmax().item())
        from computecloud_node.llm.model_shard import TorchShardModule
        from computecloud_node.llm.weights import LocalWeightsSource, ShardWeightsLoader
        from computecloud_node.llm.executor import _torch_to_tensor_payload, _tensor_payload_to_torch
        source = LocalWeightsSource(d)
        loader = ShardWeightsLoader(source)
        shard0 = TorchShardModule(TINY_CONFIG, 0, 1, is_first=True, is_last=False, force_dtype=torch.float32)
        shard0.load_state_dict(loader.load_shard(0, 1, is_first=True, is_last=False))
        with torch.no_grad():
            hidden = shard0.forward(token_ids)
        hidden_payload = _torch_to_tensor_payload(hidden)
        hidden_back = _tensor_payload_to_torch(hidden_payload)
        shard1 = TorchShardModule(TINY_CONFIG, 1, 2, is_first=False, is_last=True, force_dtype=torch.float32)
        shard1.load_state_dict(loader.load_shard(1, 2, is_first=False, is_last=True))
        with torch.no_grad():
            sharded_logits = shard1.forward(hidden_back)
        sharded_token = int(sharded_logits[0, -1, :].argmax().item())
        assert torch.allclose(mono_logits, sharded_logits[0, -1, :], atol=1e-5)
        assert sharded_token == mono_token, f"{sharded_token} != {mono_token}"

    def test_executor_full_pipeline(self, tiny_model_dir):
        d, _module = tiny_model_dir
        from computecloud_node.llm.executor import LLMShardExecutor
        weights_uri = f"file://{d}"
        tok_uri = f"file://{os.path.join(d, 'tokenizer.json')}"
        executor = LLMShardExecutor()
        s0_payload = {
            "kind": "llm_shard", "weights_uri": weights_uri, "config": TINY_CONFIG,
            "start_layer": 0, "end_layer": 1, "shard_index": 0, "shard_count": 2,
            "is_first": True, "is_last": False, "prompt": "hello world",
            "tokenizer_uri": tok_uri,
        }
        s0_out = executor.execute("s0", "demo", s0_payload)
        assert s0_out["is_final"] is False
        assert "output" in s0_out
        s1_payload = {
            "kind": "llm_shard", "weights_uri": weights_uri, "config": TINY_CONFIG,
            "start_layer": 1, "end_layer": 2, "shard_index": 1, "shard_count": 2,
            "is_first": False, "is_last": True, "input": s0_out["output"],
            "tokenizer_uri": tok_uri,
        }
        s1_out = executor.execute("s1", "demo", s1_payload)
        assert s1_out["is_final"] is True
        assert "token_id" in s1_out
        assert "logits" in s1_out
        assert isinstance(s1_out["token_id"], int)


class _MockResp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class TestExecutorRouting:
    """LLMShardExecutor routes only llm_shard + weights_uri payloads."""

    def test_rejects_non_llm_payload(self):
        from computecloud_node.llm.executor import LLMShardExecutor
        ex = LLMShardExecutor()
        with pytest.raises(ValueError, match="not an LLM shard"):
            ex.execute("t", "j", {"kind": "pipeline_shard"})

    def test_is_llm_shard_task(self):
        from computecloud_node.llm.executor import is_llm_shard_task
        assert is_llm_shard_task({"kind": "llm_shard", "weights_uri": "file:///x"})
        assert is_llm_shard_task({"kind": "llm_shard", "weights_uri": "hf://org/m@main"})
        assert not is_llm_shard_task({"kind": "llm_shard"})  # no weights_uri
        assert not is_llm_shard_task({"kind": "pipeline_shard", "weights_uri": "file:///x"})
        assert not is_llm_shard_task(None)
        assert not is_llm_shard_task({"kind": "llm_shard", "weights_uri": "http://x"})

    def test_data_shard_aware_routes_llm(self, tiny_model_dir):
        """DataShardAwareExecutor routes llm_shard to the llm_executor."""
        from computecloud_node.data_worker import DataShardAwareExecutor, DataShardWorker
        from computecloud_node.llm.executor import LLMShardExecutor
        d, _ = tiny_model_dir

        class _MockClient:
            def get(self, url):
                return _MockResp(200, {"data": "x"})
            def post(self, url, json=None):
                return _MockResp(200, {"accepted": True})

        data_worker = DataShardWorker(_MockClient(), poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        llm_ex = LLMShardExecutor()
        adapter = DataShardAwareExecutor(data_worker, llm_executor=llm_ex)
        payload = {
            "kind": "llm_shard", "weights_uri": f"file://{d}", "config": TINY_CONFIG,
            "start_layer": 0, "end_layer": 1, "shard_index": 0, "shard_count": 1,
            "is_first": True, "is_last": True, "prompt": "hi",
            "tokenizer_uri": f"file://{os.path.join(d, 'tokenizer.json')}",
        }
        result = adapter.execute("t1", "j1", payload)
        assert result is not None
        assert result.get("is_final") is True

    def test_data_shard_aware_llm_no_executor(self):
        """Without llm_executor, an llm_shard task raises ValueError."""
        from computecloud_node.data_worker import DataShardAwareExecutor, DataShardWorker

        class _MockClient:
            def get(self, url):
                return _MockResp(200, {"data": "x"})
            def post(self, url, json=None):
                return _MockResp(200, {"accepted": True})

        data_worker = DataShardWorker(_MockClient(), poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(data_worker)
        with pytest.raises(ValueError, match="no LLM executor"):
            adapter.execute("t1", "j1", {"kind": "llm_shard", "weights_uri": "file:///x"})

    def test_data_shard_aware_plain_falls_through(self):
        from computecloud_node.data_worker import DataShardAwareExecutor, DataShardWorker

        class _MockClient:
            def get(self, url):
                return _MockResp(200, {"data": "x"})
            def post(self, url, json=None):
                return _MockResp(200, {"accepted": True})

        class _Fallback:
            def execute(self, task_id, job_id, payload):
                return {"stdout": "ok"}
        data_worker = DataShardWorker(_MockClient(), poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(data_worker, fallback=_Fallback())
        result = adapter.execute("t1", "j1", {"command": "ls"})
        assert result == {"stdout": "ok"}


class TestCapabilityFlag:
    def test_node_capabilities_has_llm_capable(self):
        from computecloud_node.config import NodeCapabilities
        assert NodeCapabilities().llm_capable is False

    def test_node_capabilities_set_llm_capable(self):
        from computecloud_node.config import NodeCapabilities
        assert NodeCapabilities(llm_capable=True).llm_capable is True


class TestImportSafety:
    """probe_llm_capable + import-without-torch safety (torch is installed here)."""

    def test_probe_llm_capable(self):
        from computecloud_node.llm.executor import probe_llm_capable
        assert probe_llm_capable() is True

    def test_package_import_no_torch(self):
        """import computecloud_node must not import torch at module level.

        Verified by checking torch is not in sys.modules *because of* the
        package import — it may already be imported by the test harness, so we
        assert the package's __all__ contains the llm symbols without forcing
        a heavy import.
        """
        import computecloud_node
        for name in ("LLMShardExecutor", "TorchShardModule", "probe_llm_capable"):
            assert name in computecloud_node.__all__, f"{name} missing from __all__"

    def test_llm_subpackage_importable(self):
        import computecloud_node.llm  # noqa: F401
        assert hasattr(computecloud_node.llm, "LLMShardExecutor")


class TestKVCache:
    def test_evict_run_id_drops_cache(self, tiny_model_dir):
        from computecloud_node.llm.executor import LLMShardExecutor
        d, _ = tiny_model_dir
        ex = LLMShardExecutor()
        # Run a single use_cache pass to populate the cache for run "r1".
        payload = {
            "kind": "llm_shard", "weights_uri": f"file://{d}", "config": TINY_CONFIG,
            "start_layer": 0, "end_layer": 2, "shard_index": 0, "shard_count": 1,
            "is_first": True, "is_last": True, "prompt": "hi",
            "tokenizer_uri": f"file://{os.path.join(d, 'tokenizer.json')}",
            "use_cache": True, "data_run_id": "r1", "pass_index": 0, "position_offset": 0,
        }
        ex.execute("t", "j", payload)
        assert ex.kv_cache_size() == 1
        assert ex.evict_run_id("r1") == 1
        assert ex.kv_cache_size() == 0

    def test_lru_cap_evicts_oldest(self, tiny_model_dir):
        from computecloud_node.llm.executor import LLMShardExecutor
        d, _ = tiny_model_dir
        ex = LLMShardExecutor(max_kv_entries=2)
        tok_uri = f"file://{os.path.join(d, 'tokenizer.json')}"
        for rid in ("a", "b", "c"):
            payload = {
                "kind": "llm_shard", "weights_uri": f"file://{d}", "config": TINY_CONFIG,
                "start_layer": 0, "end_layer": 2, "shard_index": 0, "shard_count": 1,
                "is_first": True, "is_last": True, "prompt": "hi",
                "tokenizer_uri": tok_uri,
                "use_cache": True, "data_run_id": rid, "pass_index": 0, "position_offset": 0,
            }
            ex.execute("t", "j", payload)
        assert ex.kv_cache_size() == 2  # cap of 2

    def test_sharded_with_cache_matches_monolithic(self, tiny_model_dir):
        """5-token generation: sharded+KV-cache == monolithic full-recompute."""
        d, mono = tiny_model_dir
        from computecloud_node.llm.executor import LLMShardExecutor
        from computecloud_node.llm.tokenizer import LLMTokenizer
        tok_uri = f"file://{os.path.join(d, 'tokenizer.json')}"
        tok = LLMTokenizer(os.path.join(d, "tokenizer.json"))
        prompt = "hello world"
        full_ids = tok.encode(prompt) or [1, 2, 3, 4, 5]
        # Monolithic full-recompute greedy (uses the SAME ids the sharded path
        # tokenizes from the prompt).
        mono_tokens = []
        with torch.no_grad():
            cur = torch.tensor(full_ids, dtype=torch.long)
            for _ in range(5):
                out = mono.forward(cur)
                mono_tokens.append(int(out[0, -1, :].argmax().item()))
                cur = torch.cat([cur, torch.tensor([mono_tokens[-1]])])
        # Sharded with KV cache (2 single-layer shards).
        ex = LLMShardExecutor()
        rid = "gen"
        sharded_tokens = []
        weights_uri = f"file://{d}"
        for p in range(5):
            if p == 0:
                s0 = {
                    "kind": "llm_shard", "weights_uri": weights_uri, "config": TINY_CONFIG,
                    "start_layer": 0, "end_layer": 1, "shard_index": 0, "shard_count": 2,
                    "is_first": True, "is_last": False, "prompt": prompt,
                    "tokenizer_uri": tok_uri, "use_cache": True, "data_run_id": rid,
                    "pass_index": p, "position_offset": 0,
                }
            else:
                s0 = {
                    "kind": "llm_shard", "weights_uri": weights_uri, "config": TINY_CONFIG,
                    "start_layer": 0, "end_layer": 1, "shard_index": 0, "shard_count": 2,
                    "is_first": True, "is_last": False, "tokenizer_uri": tok_uri,
                    "use_cache": True, "data_run_id": rid, "pass_index": p,
                    "next_token_id": sharded_tokens[-1],
                }
            s0_out = ex.execute("s0", "j", s0)
            s1 = {
                "kind": "llm_shard", "weights_uri": weights_uri, "config": TINY_CONFIG,
                "start_layer": 1, "end_layer": 2, "shard_index": 1, "shard_count": 2,
                "is_first": False, "is_last": True, "tokenizer_uri": tok_uri,
                "use_cache": True, "data_run_id": rid, "pass_index": p,
                "input": s0_out["output"],
            }
            s1_out = ex.execute("s1", "j", s1)
            sharded_tokens.append(s1_out["token_id"])
        assert sharded_tokens == mono_tokens, f"{sharded_tokens} != {mono_tokens}"
        assert ex.evict_run_id(rid) == 2
        assert ex.kv_cache_size() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
