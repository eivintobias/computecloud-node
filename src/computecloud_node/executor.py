"""Task executor interface and result model for the Node Client.

Users implement :class:`TaskExecutor` to define what their worker does
when it receives a task from the pool.  The
:class:`~computecloud_node.node.ComputeNode` calls ``execute`` in a
thread and reports the returned :class:`TaskResult` back to the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


@dataclass
class TaskResult:
    """Outcome of executing a single task.

    Attributes
    ----------
    success:
        Whether the task completed without error.
    result:
        Arbitrary JSON-serialisable output produced by the task.
    error:
        Error message if ``success`` is ``False``.
    execution_time_seconds:
        How long the task took to execute.
    """

    success: bool = True
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_seconds: float = 0.0
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@runtime_checkable
class TaskExecutor(Protocol):
    """Protocol that user-provided executors must implement.

    The :class:`~computecloud_node.node.ComputeNode` calls ``execute``
    in a worker thread.  A return value of ``None`` is treated as a
    successful task with no structured result.
    """

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run the task described by *payload* and return its result.

        Parameters
        ----------
        task_id:
            Unique identifier for this task.
        job_id:
            Identifier of the parent job.
        payload:
            Task payload (may be ``None`` if the scheduler sent none).

        Returns
        -------
        A result dict, or ``None`` to signal success with no output.
        """
        ...
