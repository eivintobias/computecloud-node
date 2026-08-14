"""Node-side generalized data shard worker (Phase 13c, ported to node_client v0.4.0).

A node that receives a data-shard task (payload ``kind == "data_shard"`` from
:class:`~the server-side UnifiedCoordinator.UnifiedCoordinator`) runs this
flow:

1. poll the server's data endpoint until this shard's input is ready,
2. execute the shard via the node's configured executor,
3. post the output back,
4. return the shard output as the task result.

For merge tasks (``kind == "data_merge"``) the worker polls for the merge input
(the collected shard outputs), applies a simple merge function, and posts the
result.  No direct node-to-node connections — NAT-proof by construction.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

from computecloud_node.pipeline_executor import PipelineShardExecutor

logger = logging.getLogger(__name__)

DATA_SHARD_KIND = "data_shard"
DATA_MERGE_KIND = "data_merge"


def is_data_shard_task(payload: dict[str, Any] | None) -> bool:
    """Return True if *payload* is a coordinator-produced data shard/merge task."""
    if not isinstance(payload, dict):
        return False
    return payload.get("kind") in (DATA_SHARD_KIND, DATA_MERGE_KIND)


class DataShardWorker:
    """Executes one data shard (or merge) task by polling/posting to the exchange.

    Parameters
    ----------
    http_client:
        An httpx.Client (or any object with ``get``/``post``).
    poll_interval_seconds:
        How long to wait between polls when the input isn't ready.
    poll_timeout_seconds:
        Maximum total time to wait for the input to become ready.
    executor:
        Optional fallback executor for command-based shards.
    """

    def __init__(
        self,
        http_client: Any,
        *,
        poll_interval_seconds: float = 0.2,
        poll_timeout_seconds: float = 60.0,
        executor: Any | None = None,
    ) -> None:
        self._http = http_client
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._executor = executor
        self._model_executor = PipelineShardExecutor()

    def execute_shard(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one shard or merge task: poll -> compute -> post -> return output."""
        run_id = payload["data_run_id"]
        shard_index = int(payload["shard_index"])
        shard_count = int(payload.get("shard_count", 0))
        topology = payload.get("topology", "parallel")
        is_merge = payload.get("kind") == DATA_MERGE_KIND
        input_data = self._poll_for_input(run_id, shard_index)
        if is_merge:
            output = self._compute_merge(input_data, payload)
        else:
            output = self._compute_shard(input_data, payload)
        self._post_output(run_id, shard_index, output, is_merge)
        return {
            "output": output, "shard_index": shard_index,
            "shard_count": shard_count, "topology": topology, "is_merge": is_merge,
        }

    # -- Internal helpers --------------------------------------------------

    def _poll_for_input(self, run_id: str, shard_index: int) -> Any:
        url = f"/api/v1/data/{run_id}/shard/{shard_index}"
        deadline = time.monotonic() + self._poll_timeout
        last_status = None
        while time.monotonic() < deadline:
            try:
                resp = self._http.get(url)
            except Exception as exc:
                logger.warning("DataShardWorker: poll failed: %s", exc)
                time.sleep(self._poll_interval)
                continue
            last_status = resp.status_code
            if resp.status_code == 200:
                return resp.json().get("data")
            time.sleep(self._poll_interval)
        raise TimeoutError(
            f"DataShardWorker: input for run {run_id[:8]} shard {shard_index} "
            f"never became ready within {self._poll_timeout}s (last {last_status})"
        )

    def _post_output(self, run_id: str, shard_index: int, output: Any, is_merge: bool) -> None:
        url = f"/api/v1/data/{run_id}/shard/{shard_index}"
        body = {"data": output}
        attempts = 3
        last_exc: Exception | None = None
        last_status = None
        for _ in range(attempts):
            try:
                resp = self._http.post(url, json=body)
                last_status = resp.status_code
                if resp.status_code in (200, 201, 204):
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.1)
        if last_exc is not None:
            raise RuntimeError(f"DataShardWorker: post failed: {last_exc}")
        raise RuntimeError(f"DataShardWorker: post returned {last_status}")

    def _compute_shard(self, input_data: Any, payload: dict[str, Any]) -> Any:
        """Execute the shard based on its type."""
        spec = payload.get("shard_spec") or {}
        workload_type = payload.get("workload_type", "")
        # Model shards: use the toy transform.
        if workload_type == "model_sharding" or "start_layer" in spec:
            step_payload = dict(spec)
            step_payload["shard_index"] = payload.get("shard_index", 0)
            step_payload["shard_count"] = payload.get("shard_count", 1)
            step_payload["input_activations"] = (
                list(input_data) if isinstance(input_data, list) else [int(input_data)]
            )
            out = self._model_executor.execute(
                task_id=f"data-shard-{payload.get('shard_index', 0)}",
                job_id=payload.get("job_id", "data-run"),
                payload=step_payload,
            )
            if out and "output_activations" in out:
                return out["output_activations"]
            return input_data
        # Command shards: materialize input to temp file, run command.
        command = spec.get("command") or payload.get("command")
        if command and self._executor is not None:
            return self._run_command_shard(command, input_data, payload)
        # Default: identity transform (echo input — for testing/demos).
        return input_data

    def _compute_merge(self, input_data: Any, payload: dict[str, Any]) -> Any:
        """Apply a merge function to the collected shard outputs."""
        if not isinstance(input_data, list):
            return input_data
        if all(isinstance(x, list) for x in input_data):
            merged: list[Any] = []
            for x in input_data:
                merged.extend(x)
            return merged
        if all(isinstance(x, (int, float)) for x in input_data):
            return sum(input_data)
        if all(isinstance(x, str) for x in input_data):
            return "\n".join(input_data)
        return input_data

    def _run_command_shard(self, command: str, input_data: Any, payload: dict[str, Any]) -> str:
        """Materialize input to a temp file and run the command."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            if isinstance(input_data, str):
                f.write(input_data)
            elif isinstance(input_data, list):
                f.write("\n".join(str(x) for x in input_data))
            else:
                f.write(str(input_data))
            input_path = f.name
        try:
            cmd = command.replace("{input_file}", input_path)
            result = self._executor.execute(
                task_id=f"data-shard-{payload.get('shard_index', 0)}",
                job_id=payload.get("job_id", "data-run"),
                payload={"command": cmd},
            )
            if result and isinstance(result, dict):
                return result.get("stdout", str(result))
            return str(result)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass


__all__ = [
    "DataShardWorker", "DataShardAwareExecutor",
    "is_data_shard_task", "DATA_SHARD_KIND", "DATA_MERGE_KIND",
]


class DataShardAwareExecutor:
    """A :class:`TaskExecutor` that routes data shards, pipeline shards, and plain tasks.

    Sibling of :class:`~computecloud_node.shard_executor_adapter.ShardAwareExecutor`
    that also handles Phase 13c data-shard tasks.  When a task payload is a
    data shard/merge (``kind == "data_shard"`` / ``"data_merge"``), the adapter
    delegates to the :class:`DataShardWorker`; when it is a pipeline shard
    (``kind == "pipeline_shard"``), it delegates to the optional
    :class:`PipelineShardWorker`; otherwise it falls back to the user-supplied
    executor.  This does NOT modify the existing ``ShardAwareExecutor`` —
    v0.3.0 pipeline routing is unaffected.

    Parameters
    ----------
    data_worker:
        The :class:`DataShardWorker` used for data shard/merge tasks.
    pipeline_worker:
        Optional :class:`PipelineShardWorker` for legacy pipeline shard tasks.
    fallback:
        Optional executor for non-shard tasks.  When ``None``, non-shard tasks
        raise ``ValueError``.
    """

    def __init__(
        self,
        data_worker: DataShardWorker,
        pipeline_worker: Any | None = None,
        fallback: Any | None = None,
    ) -> None:
        self._data_worker = data_worker
        self._pipeline_worker = pipeline_worker
        self._fallback = fallback

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if is_data_shard_task(payload):
            assert payload is not None
            return self._data_worker.execute_shard(payload)
        # Legacy pipeline shard tasks.
        if isinstance(payload, dict) and payload.get("kind") == "pipeline_shard":
            if self._pipeline_worker is not None:
                return self._pipeline_worker.execute_shard(payload)
            raise ValueError(
                f"DataShardAwareExecutor: pipeline shard task {task_id} received "
                f"but no pipeline worker was configured"
            )
        if self._fallback is not None:
            return self._fallback.execute(task_id=task_id, job_id=job_id, payload=payload)
        raise ValueError(
            f"DataShardAwareExecutor: non-shard task {task_id} received but no "
            f"fallback executor was configured"
        )
