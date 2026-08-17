"""Pure-stdlib LLM support tests — run WITHOUT torch (Phase 16).

These tests cover the supporting pieces that have NO torch dependency:
``computecloud_node.tensor_format``, ``computecloud_node.hf_uri``,
``computecloud_node.llm.uri`` (file:// URI parsing), ``keys_for_shard``
(filtering logic), the ``NodeCapabilities.llm_capable`` flag, and the
routing predicates (``is_llm_shard_task`` / ``is_data_shard_task``).

They deliberately do **not** ``importorskip("torch")`` so they stay green
when the ``[llm]`` extra is absent — verifying the non-torch subset of the
node_client suite passes with torch absent (simulated by simply not having
the extra).
"""

from __future__ import annotations

import pytest


class TestTensorPayloadStdlib:
    def test_float32_roundtrip(self):
        from computecloud_node.tensor_format import TensorPayload
        data = [[1.0, 2.0], [3.0, 4.0]]
        p = TensorPayload.from_nested_list(data, "float32")
        assert p.shape == (2, 2)
        assert p.to_nested_list() == data

    def test_int64_roundtrip(self):
        from computecloud_node.tensor_format import TensorPayload
        p = TensorPayload.from_nested_list([[1, 2, 3]], "int64")
        assert p.to_nested_list() == [[1, 2, 3]]

    def test_unknown_dtype_rejected(self):
        from computecloud_node.tensor_format import TensorPayload
        with pytest.raises(ValueError, match="unknown dtype"):
            TensorPayload.from_nested_list([[1.0]], "bogus")

    def test_byte_length_mismatch(self):
        from computecloud_node.tensor_format import TensorPayload
        with pytest.raises(ValueError, match="byte length mismatch"):
            TensorPayload(shape=(4,), dtype="float32", data_b64="AAAA").validate()

    def test_to_json_from_json(self):
        from computecloud_node.tensor_format import TensorPayload
        p = TensorPayload.from_nested_list([[1.0, 2.0]], "float32")
        p2 = TensorPayload.from_json(p.to_json())
        assert p2.to_nested_list() == [[1.0, 2.0]]


class TestHfUriStdlib:
    def test_parse_with_revision(self):
        from computecloud_node.hf_uri import parse_hf_uri
        assert parse_hf_uri("hf://org/model@v1.0") == ("org/model", "v1.0")

    def test_parse_without_revision(self):
        from computecloud_node.hf_uri import parse_hf_uri
        assert parse_hf_uri("hf://org/model") == ("org/model", "main")

    def test_reject_traversal(self):
        from computecloud_node.hf_uri import parse_hf_uri
        with pytest.raises(ValueError, match=r"\.\."):
            parse_hf_uri("hf://org/../model@main")

    def test_build_roundtrip(self):
        from computecloud_node.hf_uri import build_hf_uri, parse_hf_uri
        assert parse_hf_uri(build_hf_uri("org/model", "main")) == ("org/model", "main")


class TestFileUriStdlib:
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

    def test_is_llm_weights_uri(self):
        from computecloud_node.llm.uri import is_llm_weights_uri
        assert is_llm_weights_uri("hf://org/model@main")
        assert is_llm_weights_uri("file:///tmp/weights")
        assert not is_llm_weights_uri("http://example.com")
        assert not is_llm_weights_uri(None)


ALL_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.input_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]


class TestKeysForShardStdlib:
    def test_first_shard_includes_embed(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 0, 1, is_first=True, is_last=False)
        assert "model.embed_tokens.weight" in keys
        assert "lm_head.weight" not in keys

    def test_last_shard_includes_lm_head(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 1, 2, is_first=False, is_last=True)
        assert "lm_head.weight" in keys
        assert "model.embed_tokens.weight" not in keys

    def test_monolithic(self):
        from computecloud_node.llm.weights import keys_for_shard
        keys = keys_for_shard(ALL_KEYS, 0, 2, is_first=True, is_last=True)
        assert len(keys) == len(ALL_KEYS)


class TestCapabilityFlagStdlib:
    def test_node_capabilities_has_llm_capable(self):
        from computecloud_node.config import NodeCapabilities
        assert NodeCapabilities().llm_capable is False

    def test_node_capabilities_set_llm_capable(self):
        from computecloud_node.config import NodeCapabilities
        assert NodeCapabilities(llm_capable=True).llm_capable is True


class TestRoutingPredicatesStdlib:
    """is_llm_shard_task + is_data_shard_task are pure stdlib (no torch)."""

    def test_is_llm_shard_task(self):
        from computecloud_node.llm.executor import is_llm_shard_task
        assert is_llm_shard_task({"kind": "llm_shard", "weights_uri": "file:///x"})
        assert is_llm_shard_task({"kind": "llm_shard", "weights_uri": "hf://org/m@main"})
        assert not is_llm_shard_task({"kind": "llm_shard"})  # no weights_uri
        assert not is_llm_shard_task({"kind": "pipeline_shard", "weights_uri": "file:///x"})
        assert not is_llm_shard_task(None)
        assert not is_llm_shard_task({"kind": "llm_shard", "weights_uri": "http://x"})

    def test_is_data_shard_task(self):
        from computecloud_node.data_worker import DATA_SHARD_KIND, is_data_shard_task
        assert is_data_shard_task({"kind": DATA_SHARD_KIND})
        assert not is_data_shard_task({"kind": "llm_shard"})
        assert not is_data_shard_task(None)


class TestPackageExportsStdlib:
    """import computecloud_node works without torch; llm symbols in __all__."""

    def test_import_computecloud_node(self):
        import computecloud_node
        assert hasattr(computecloud_node, "ComputeNode")
        assert hasattr(computecloud_node, "NodeConfig")

    def test_llm_exports_in_all(self):
        import computecloud_node
        for name in ("LLMShardExecutor", "TorchShardModule", "probe_llm_capable",
                     "is_llm_shard_task"):
            assert name in computecloud_node.__all__, f"{name} missing from __all__"

    def test_individual_modules_importable(self):
        import computecloud_node.hf_uri  # noqa: F401
        import computecloud_node.llm.executor  # noqa: F401
        import computecloud_node.llm.uri  # noqa: F401
        import computecloud_node.llm.weights  # noqa: F401
        import computecloud_node.tensor_format  # noqa: F401
