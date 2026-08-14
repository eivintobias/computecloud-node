"""Tests for the ported data_worker and DataShardAwareExecutor (Phase 14b).

Verifies is_data_shard_task, DataShardWorker (poll/compute/post with mock HTTP
client; identity/model/merge shards; poll timeout), and DataShardAwareExecutor
routing (data shard, pipeline shard, plain task fall-through) — identical logic
to the in-repo tests but with computecloud_node imports.
"""

from __future__ import annotations

import pytest

from computecloud_node.data_worker import (
    DATA_MERGE_KIND,
    DATA_SHARD_KIND,
    DataShardAwareExecutor,
    DataShardWorker,
    is_data_shard_task,
)


class TestIsDataShardTask:
    def test_none_payload(self):
        assert is_data_shard_task(None) is False

    def test_non_dict(self):
        assert is_data_shard_task("nope") is False
        assert is_data_shard_task(42) is False

    def test_data_shard(self):
        assert is_data_shard_task({"kind": DATA_SHARD_KIND}) is True

    def test_data_merge(self):
        assert is_data_shard_task({"kind": DATA_MERGE_KIND}) is True

    def test_pipeline_shard_not_data(self):
        assert is_data_shard_task({"kind": "pipeline_shard"}) is False

    def test_plain_task(self):
        assert is_data_shard_task({"command": "ls"}) is False


class _MockResp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class _MockClient:
    def __init__(self, data=None, post_ok=True):
        self._data = data
        self._post_ok = post_ok
        self.posted = []

    def get(self, url):
        if self._data is None:
            return _MockResp(204)
        return _MockResp(200, {"data": self._data})

    def post(self, url, json=None):
        self.posted.append(json)
        if not self._post_ok:
            return _MockResp(409)
        return _MockResp(200, {"accepted": True})


class TestDataShardWorker:
    def test_identity_shard(self):
        client = _MockClient(data="hello")
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        payload = {
            "kind": DATA_SHARD_KIND, "data_run_id": "r1",
            "shard_index": 0, "shard_count": 1, "topology": "parallel",
            "shard_spec": {}, "workload_type": "rendering",
        }
        out = worker.execute_shard(payload)
        assert out["shard_index"] == 0
        assert out["is_merge"] is False
        assert out["output"] == "hello"
        assert len(client.posted) == 1

    def test_model_shard_uses_toy_transform(self):
        client = _MockClient(data=[1, 2, 3])
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        payload = {
            "kind": DATA_SHARD_KIND, "data_run_id": "r1",
            "shard_index": 0, "shard_count": 1, "topology": "pipeline",
            "shard_spec": {"model_name": "t", "start_layer": 0, "end_layer": 4},
            "workload_type": "model_sharding",
        }
        out = worker.execute_shard(payload)
        assert isinstance(out["output"], list)
        assert len(out["output"]) == 3

    def test_merge_shard(self):
        client = _MockClient(data=[[1, 2], [3, 4]])
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        payload = {
            "kind": DATA_MERGE_KIND, "data_run_id": "r1",
            "shard_index": 2, "shard_count": 2, "topology": "reduce",
            "shard_spec": {"is_merge": True}, "workload_type": "batch",
        }
        out = worker.execute_shard(payload)
        assert out["is_merge"] is True
        assert out["output"] == [1, 2, 3, 4]

    def test_poll_timeout_raises(self):
        client = _MockClient(data=None)
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=0.3)
        payload = {
            "kind": DATA_SHARD_KIND, "data_run_id": "r1",
            "shard_index": 0, "shard_count": 1, "topology": "parallel",
            "shard_spec": {}, "workload_type": "rendering",
        }
        with pytest.raises(TimeoutError):
            worker.execute_shard(payload)

    def test_post_failure_raises(self):
        client = _MockClient(data="x", post_ok=False)
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        payload = {
            "kind": DATA_SHARD_KIND, "data_run_id": "r1",
            "shard_index": 0, "shard_count": 1, "topology": "parallel",
            "shard_spec": {}, "workload_type": "rendering",
        }
        with pytest.raises(RuntimeError, match="post"):
            worker.execute_shard(payload)


class TestDataShardAwareExecutor:
    def test_routes_data_shard(self):
        client = _MockClient(data="x")
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(worker)
        result = adapter.execute("t1", "j1", {"kind": DATA_SHARD_KIND, "data_run_id": "r", "shard_index": 0, "shard_count": 1, "topology": "parallel", "shard_spec": {}, "workload_type": "rendering"})
        assert result is not None
        assert result["output"] == "x"

    def test_routes_pipeline_shard(self):
        from computecloud_node.pipeline_worker import PipelineShardWorker

        class _PipeMockClient:
            def __init__(self):
                self.posted = []
            def get(self, url):
                return _MockResp(200, {"activations": [1, 2, 3]})
            def post(self, url, json=None):
                self.posted.append(json)
                return _MockResp(200, {"accepted": True})

        client = _PipeMockClient()
        data_worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        pipe_worker = PipelineShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(data_worker, pipeline_worker=pipe_worker)
        result = adapter.execute("t1", "j1", {
            "kind": "pipeline_shard", "pipeline_run_id": "r",
            "shard_index": 0, "shard_count": 1,
            "shard_spec": {"model_name": "t", "start_layer": 0, "end_layer": 4},
            "job_id": "j",
        })
        assert result is not None
        assert "output_activations" in result

    def test_plain_task_falls_through(self):
        class _Fallback:
            def execute(self, task_id, job_id, payload):
                return {"stdout": "ok"}
        client = _MockClient(data="x")
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(worker, fallback=_Fallback())
        result = adapter.execute("t1", "j1", {"command": "echo hello"})
        assert result == {"stdout": "ok"}

    def test_no_fallback_raises(self):
        client = _MockClient(data="x")
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(worker)
        with pytest.raises(ValueError, match="no fallback"):
            adapter.execute("t1", "j1", {"command": "ls"})

    def test_pipeline_shard_without_pipeline_worker_raises(self):
        client = _MockClient(data="x")
        worker = DataShardWorker(client, poll_interval_seconds=0.001, poll_timeout_seconds=2.0)
        adapter = DataShardAwareExecutor(worker)
        with pytest.raises(ValueError, match="no pipeline worker"):
            adapter.execute("t1", "j1", {"kind": "pipeline_shard", "pipeline_run_id": "r", "shard_index": 0, "shard_count": 1, "shard_spec": {}})
