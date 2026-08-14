"""Tests for the ported pipeline_executor module (Phase 12).

Verifies PipelineShardExecutor, LocalPipelineRunner, and run_reference
work correctly in the standalone package — identical logic to the in-repo
tests but with computecloud_node imports.
"""

from __future__ import annotations

import pytest

from computecloud_node.pipeline_executor import (
    LocalPipelineRunner,
    PipelineShardExecutor,
    run_reference,
)


def _shard_payload(model, start, end, idx, count):
    return {
        "model_name": model,
        "start_layer": start,
        "end_layer": end,
        "shard_index": idx,
        "shard_count": count,
    }


class TestRunReference:
    def test_single_layer(self):
        out = run_reference("toy", 1, [1, 2, 3])
        assert isinstance(out, list)
        assert len(out) == 3

    def test_deterministic(self):
        a = run_reference("toy-24l", 24, [1, 2, 3])
        b = run_reference("toy-24l", 24, [1, 2, 3])
        assert a == b

    def test_different_models_differ(self):
        a = run_reference("model-A", 10, [1, 2])
        b = run_reference("model-B", 10, [1, 2])
        assert a != b


class TestPipelineShardExecutor:
    def test_execute_matches_reference(self):
        model = "toy-24l"
        total = 24
        payload = _shard_payload(model, 0, total, 0, 1)
        payload["input_activations"] = [1, 2, 3]
        result = PipelineShardExecutor().execute("t1", "j1", payload)
        ref = run_reference(model, total, [1, 2, 3])
        assert result["output_activations"] == ref
        assert result["is_final"] is True

    def test_is_final_only_on_last_shard(self):
        executor = PipelineShardExecutor()
        for i in range(3):
            p = _shard_payload("m", 0, 1, i, 3)
            p["input_activations"] = [1]
            r = executor.execute("t", "j", p)
            assert r["is_final"] == (i == 2)

    def test_none_payload_rejected(self):
        with pytest.raises(ValueError, match="payload"):
            PipelineShardExecutor().execute("t", "j", None)

    def test_missing_model_name_rejected(self):
        payload = {"start_layer": 0, "end_layer": 1, "shard_index": 0,
                    "shard_count": 1, "input_activations": [1]}
        with pytest.raises(ValueError, match="model_name"):
            PipelineShardExecutor().execute("t", "j", payload)

    def test_missing_input_activations_rejected(self):
        payload = _shard_payload("m", 0, 1, 0, 1)
        with pytest.raises(ValueError, match="input_activations"):
            PipelineShardExecutor().execute("t", "j", payload)


class TestLocalPipelineRunner:
    def test_chain_matches_reference(self):
        model = "toy-24l"
        total = 24
        shards = [
            _shard_payload(model, 0, 8, 0, 3),
            _shard_payload(model, 8, 16, 1, 3),
            _shard_payload(model, 16, 24, 2, 3),
        ]
        runner = LocalPipelineRunner()
        result = runner.run(shards, [1, 2, 3])
        ref = run_reference(model, total, [1, 2, 3])
        assert result["output_activations"] == ref
        assert result["is_final"] is True

    def test_empty_payloads_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            LocalPipelineRunner().run([], [1])

    def test_unordered_shards_raises(self):
        shards = [
            _shard_payload("m", 0, 1, 1, 2),  # wrong index
            _shard_payload("m", 1, 2, 0, 2),
        ]
        with pytest.raises(ValueError, match="ordered"):
            LocalPipelineRunner().run(shards, [1])
