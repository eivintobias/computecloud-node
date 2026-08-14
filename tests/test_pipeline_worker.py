"""Tests for the ported pipeline_worker and shard_executor_adapter (Phase 12).

Verifies is_shard_task, PipelineShardWorker (poll/compute/post with mock
HTTP client), and ShardAwareExecutor routing — identical logic to the
in-repo tests but with computecloud_node imports.
"""

from __future__ import annotations

import pytest

from computecloud_node.pipeline_worker import PipelineShardWorker, is_shard_task
from computecloud_node.shard_executor_adapter import ShardAwareExecutor


class _MockClient:
    """Minimal httpx-like client for testing."""

    def __init__(self, activations=None):
        self._activations = activations
        self.posted = []

    def get(self, url):
        if self._activations is None:
            return _MockResp(204)
        return _MockResp(200, {"activations": self._activations})

    def post(self, url, json=None):
        self.posted.append((url, json))
        return _MockResp(200)


class _MockResp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class TestIsShardTask:
    def test_none_payload(self):
        assert is_shard_task(None) is False

    def test_non_dict_payload(self):
        assert is_shard_task("not a dict") is False
        assert is_shard_task(42) is False

    def test_shard_task(self):
        assert is_shard_task({"kind": "pipeline_shard"}) is True

    def test_non_shard_task(self):
        assert is_shard_task({"kind": "other"}) is False
        assert is_shard_task({"command": "ls"}) is False


class TestPipelineShardWorker:
    def test_execute_shard_polls_computes_posts(self):
        client = _MockClient(activations=[1, 2, 3])
        worker = PipelineShardWorker(
            client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0
        )
        payload = {
            "pipeline_run_id": "run-1",
            "shard_index": 0,
            "shard_count": 2,
            "shard_spec": {
                "model_name": "toy-24l",
                "start_layer": 0,
                "end_layer": 8,
            },
        }
        result = worker.execute_shard(payload)
        assert "output_activations" in result
        assert result["shard_index"] == 0
        assert result["is_final"] is False
        assert len(client.posted) == 1

    def test_execute_shard_final_shard_is_final(self):
        client = _MockClient(activations=[1])
        worker = PipelineShardWorker(client, poll_timeout_seconds=2.0)
        payload = {
            "pipeline_run_id": "run-1",
            "shard_index": 2,
            "shard_count": 3,
            "shard_spec": {"model_name": "m", "start_layer": 0, "end_layer": 1},
        }
        result = worker.execute_shard(payload)
        assert result["is_final"] is True

    def test_poll_timeout_raises(self):
        client = _MockClient(activations=None)  # never ready
        worker = PipelineShardWorker(
            client, poll_interval_seconds=0.01, poll_timeout_seconds=0.05
        )
        payload = {
            "pipeline_run_id": "run-x",
            "shard_index": 0,
            "shard_count": 1,
            "shard_spec": {"model_name": "m", "start_layer": 0, "end_layer": 1},
        }
        with pytest.raises(TimeoutError, match="never became ready"):
            worker.execute_shard(payload)


class TestShardAwareExecutor:
    def test_routes_shard_task_to_worker(self):
        client = _MockClient(activations=[1, 2])
        worker = PipelineShardWorker(client, poll_timeout_seconds=2.0)
        adapter = ShardAwareExecutor(worker)
        payload = {
            "kind": "pipeline_shard",
            "pipeline_run_id": "r",
            "shard_index": 0,
            "shard_count": 1,
            "shard_spec": {"model_name": "m", "start_layer": 0, "end_layer": 1},
        }
        result = adapter.execute(task_id="t", job_id="j", payload=payload)
        assert "output_activations" in result

    def test_non_shard_task_no_fallback_raises(self):
        client = _MockClient()
        worker = PipelineShardWorker(client)
        adapter = ShardAwareExecutor(worker)
        with pytest.raises(ValueError, match="no fallback"):
            adapter.execute(task_id="t", job_id="j", payload={"command": "ls"})

    def test_non_shard_task_with_fallback(self):
        from computecloud_node.executor import TaskExecutor

        class Fallback(TaskExecutor):
            def execute(self, task_id, job_id, payload):
                return {"fallback": True}

        client = _MockClient()
        worker = PipelineShardWorker(client)
        adapter = ShardAwareExecutor(worker, fallback=Fallback())
        out = adapter.execute(task_id="t", job_id="j", payload={"command": "ls"})
        assert out == {"fallback": True}
