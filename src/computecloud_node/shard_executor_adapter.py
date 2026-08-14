"""Shard-aware TaskExecutor adapter for the node client (Phase 9c-2).

Wraps a :class:`~computecloud_node.pipeline_worker.PipelineShardWorker` so a
:class:`~computecloud_node.node.ComputeNode` can handle pipeline-shard tasks
through its existing pull/report loop: when a task payload is a coordinator
shard task (``kind == "pipeline_shard"``), the adapter delegates to the
worker; otherwise it falls back to a user-supplied executor (e.g.
:class:`~computecloud_node.local_executor.LocalProcessExecutor`).

This keeps the node's runtime loop untouched -- the adapter is just a
:class:`~computecloud_node.executor.TaskExecutor` implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from computecloud_node.executor import TaskExecutor
from computecloud_node.pipeline_worker import PipelineShardWorker, is_shard_task

logger = logging.getLogger(__name__)


class ShardAwareExecutor:
    """A :class:`TaskExecutor` that routes shard tasks to a worker.

    Parameters
    ----------
    shard_worker:
        The :class:`PipelineShardWorker` used for ``pipeline_shard`` tasks.
    fallback:
        Optional executor for non-shard tasks.  When ``None``, non-shard tasks
        raise ``ValueError``.
    """

    def __init__(
        self,
        shard_worker: PipelineShardWorker,
        fallback: TaskExecutor | None = None,
    ) -> None:
        self._shard_worker = shard_worker
        self._fallback = fallback

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if is_shard_task(payload):
            assert payload is not None
            logger.info(
                "ShardAwareExecutor: executing shard task %s (run %s shard %d)",
                task_id[:8],
                str(payload.get("pipeline_run_id", ""))[:8],
                int(payload.get("shard_index", -1)),
            )
            return self._shard_worker.execute_shard(payload)
        if self._fallback is not None:
            return self._fallback.execute(task_id=task_id, job_id=job_id, payload=payload)
        raise ValueError(
            f"ShardAwareExecutor: non-shard task {task_id} received but no "
            f"fallback executor was configured"
        )


__all__ = ["ShardAwareExecutor"]
