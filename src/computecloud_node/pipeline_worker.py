"""Node-side shard worker -- executes one pipeline shard task (Phase 9c-2).

A node that receives a shard task (payload ``kind == "pipeline_shard"`` from
:class:`~the server-side PipelineCoordinator.PipelineCoordinator`) runs
this flow:

1. poll the server's activation endpoint until this shard's input is ready
   (``GET .../pipeline/{run_id}/activation/{shard_index}``),
2. compute the shard's layers via 9c-1's
   :class:`~computecloud_node.pipeline_executor.PipelineShardExecutor` (the
   toy hash-derived transform -- no torch/ML/GPU),
3. post the output back
   (``POST .../pipeline/{run_id}/activation/{shard_index}``), which the
   :class:`~the server-side ActivationExchange.ActivationExchange` turns
   into the next shard's input (or the run's final output),
4. return the shard output as the task result so the existing node
   pull/report machinery completes the task normally and the coordinator's
   cleanup fires.

This mirrors how :class:`~computecloud_node.node.ComputeNode` already talks
to the server (httpx client, bounded retries, fail-the-task on timeout).  No
direct node-to-node connections -- the server mediates, so this is
NAT-proof by construction.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from computecloud_node.pipeline_executor import PipelineShardExecutor

logger = logging.getLogger(__name__)


def is_shard_task(payload: dict[str, Any] | None) -> bool:
    """Return True if *payload* is a coordinator-produced shard task."""
    if not isinstance(payload, dict):
        return False
    return payload.get("kind") == "pipeline_shard"


class PipelineShardWorker:
    """Executes one pipeline shard task by polling/posting to the exchange.

    This is a helper used by the node's task executor when it recognises a
    shard task payload.  It does *not* own the HTTP client -- the caller passes
    an httpx-style client (the same one :class:`ComputeNode` uses for pull/
    report) so we reuse the existing transport machinery.

    Parameters
    ----------
    http_client:
        An httpx.Client (or any object with ``get``/``post`` returning a
        response with ``status_code`` and ``json()``).
    poll_interval_seconds:
        How long to wait between activation polls when the input isn't ready.
    poll_timeout_seconds:
        Maximum total time to wait for the input to become ready before
        failing the shard.
    executor:
        Optional :class:`PipelineShardExecutor` (defaults to a fresh one).
    """

    def __init__(
        self,
        http_client: Any,
        *,
        poll_interval_seconds: float = 0.2,
        poll_timeout_seconds: float = 60.0,
        executor: PipelineShardExecutor | None = None,
    ) -> None:
        self._http = http_client
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._executor: PipelineShardExecutor = executor or PipelineShardExecutor()

    def execute_shard(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one shard task: poll -> compute -> post -> return output.

        Parameters
        ----------
        payload:
            The shard task payload produced by the coordinator.  Must carry
            ``pipeline_run_id``, ``shard_index``, ``shard_count``, and
            ``shard_spec`` (the 9c-1 shard spec with ``model_name``,
            ``start_layer``, ``end_layer``).

        Returns
        -------
        The shard output dict (``output_activations``, ``shard_index``,
        ``is_final``) -- also posted to the exchange.

        Raises
        ------
        TimeoutError:
            If the input never becomes ready within ``poll_timeout_seconds``.
        """
        run_id = payload["pipeline_run_id"]
        shard_index = int(payload["shard_index"])
        shard_count = int(payload["shard_count"])
        shard_spec = dict(payload.get("shard_spec") or {})

        # 1. Poll for the input activations.
        activations = self._poll_for_input(run_id, shard_index)

        # 2. Compute the shard's layers (9c-1 toy transform).
        step_payload = dict(shard_spec)
        step_payload["input_activations"] = list(activations)
        # PipelineShardExecutor also requires shard_index/shard_count in the
        # payload (the 9c-1 LocalPipelineRunner injected these); carry them
        # through so the executor validates cleanly.
        step_payload.setdefault("shard_index", shard_index)
        step_payload.setdefault("shard_count", shard_count)
        # PipelineShardExecutor.execute ignores task_id/job_id for the math.
        out = self._executor.execute(
            task_id=f"pipeline-{run_id}-shard-{shard_index}",
            job_id=payload.get("job_id", "pipeline-run"),
            payload=step_payload,
        )
        if out is None or "output_activations" not in out:
            raise RuntimeError(
                f"PipelineShardWorker shard {shard_index} produced no output"
            )
        output_activations = list(out["output_activations"])

        # 3. Post the output back to the exchange (becomes next shard's input
        #    or the run's final output).
        self._post_output(run_id, shard_index, output_activations)

        # 4. Return the shard output as the task result.
        return {
            "output_activations": output_activations,
            "shard_index": shard_index,
            "is_final": shard_index == shard_count - 1,
        }

    # -- Internal helpers --------------------------------------------------

    def _poll_for_input(self, run_id: str, shard_index: int) -> list[int]:
        """Poll the activation endpoint until the input is ready."""
        url = f"/api/v1/pipeline/{run_id}/activation/{shard_index}"
        deadline = time.monotonic() + self._poll_timeout
        last_status = None
        while time.monotonic() < deadline:
            try:
                resp = self._http.get(url)
            except Exception as exc:
                logger.warning(
                    "PipelineShardWorker: poll failed for run %s shard %d: %s",
                    run_id[:8], shard_index, exc,
                )
                time.sleep(self._poll_interval)
                continue
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                acts = data.get("activations")
                if isinstance(acts, list):
                    return [int(v) for v in acts]
            # 204 / not-ready -> keep polling.
            time.sleep(self._poll_interval)
        raise TimeoutError(
            f"PipelineShardWorker: input for run {run_id[:8]} shard {shard_index} "
            f"never became ready within {self._poll_timeout}s "
            f"(last status {last_status})"
        )

    def _post_output(
        self, run_id: str, shard_index: int, output: list[int]
    ) -> None:
        """Post the shard output back to the exchange."""
        url = f"/api/v1/pipeline/{run_id}/activation/{shard_index}"
        body = {"activations": list(output)}
        # Bounded retries on transient failure.
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
            raise RuntimeError(
                f"PipelineShardWorker: failed to post output for run {run_id[:8]} "
                f"shard {shard_index} after {attempts} attempts: {last_exc}"
            )
        raise RuntimeError(
            f"PipelineShardWorker: post output for run {run_id[:8]} shard "
            f"{shard_index} returned {last_status}"
        )


__all__ = ["PipelineShardWorker", "is_shard_task"]

