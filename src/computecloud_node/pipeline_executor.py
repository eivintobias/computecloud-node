"""Toy pipeline executor -- local proof-of-concept for Phase 9c-1.

Executes model-shard tasks **without any ML libraries**: the "model" is a toy
deterministic transform.  Each layer ``i`` applies::

    value = (value * multipliers[i] + biases[i]) % modulus

to a list of integers (the "activations").  Layer parameters derive
deterministically from ``(model_name, layer_index)`` via :mod:`hashlib`, so
every node computes identical results with no weight files, no torch, no
transformers, and no GPU.

This module provides three things:

* :class:`PipelineShardExecutor` -- a concrete
  :class:`~computecloud_node.executor.TaskExecutor` that runs one shard's
  layers against an input activation list and returns the output list.
* :class:`LocalPipelineRunner` -- a small orchestrator that takes an ordered
  list of shard task payloads and an initial input, executes shard 0 -> feeds
  its ``output_activations`` into shard 1's payload -> ... -> returns the
  final shard's output.  Runs shards sequentially in-process.
* :func:`run_reference` -- a pure-function reference implementation that runs
  *all* layers of a model in a single pass, so tests can assert that an
  N-shard pipeline output equals the reference single-pass output.

Payload contract (per shard)::

    {
        "model_name": str,
        "start_layer": int,        # inclusive, 0-based
        "end_layer": int,          # exclusive
        "shard_index": int,
        "shard_count": int,
        "input_activations": list[int],   # required on every shard
    }

Output::

    {
        "output_activations": list[int],
        "shard_index": int,
        "is_final": bool,           # True only on the last shard
    }
"""

from __future__ import annotations

import hashlib
from typing import Any

from computecloud_node.executor import TaskExecutor

# A large prime modulus keeps the toy activations in a bounded range and
# makes the transform non-trivial (multiplication + addition + wraparound).
_MODULUS = 1_000_000_007


def _layer_params(model_name: str, layer_index: int) -> tuple[int, int]:
    """Derive (multiplier, bias) for one layer deterministically from hashlib.

    Uses SHA-256 of ``f"{model_name}:{layer_index}"`` so every node computes
    identical layer parameters with no weight files.  The multiplier is kept
    in ``[1, modulus)`` (never zero, so the transform is invertible) and the
    bias in ``[0, modulus)``.
    """
    digest = hashlib.sha256(f"{model_name}:{layer_index}".encode()).digest()
    mult = int.from_bytes(digest[:8], "big") % (_MODULUS - 1) + 1
    bias = int.from_bytes(digest[8:16], "big") % _MODULUS
    return mult, bias


def _apply_layer(values: list[int], mult: int, bias: int) -> list[int]:
    """Apply ``value = (value * mult + bias) % modulus`` element-wise."""
    return [((v * mult + bias) % _MODULUS) for v in values]


def _run_layers(
    model_name: str, start_layer: int, end_layer: int, values: list[int]
) -> list[int]:
    """Run layers ``[start, end)`` of *model_name* over *values* in order."""
    out = list(values)
    for i in range(start_layer, end_layer):
        mult, bias = _layer_params(model_name, i)
        out = _apply_layer(out, mult, bias)
    return out


def run_reference(
    model_name: str, total_layers: int, values: list[int]
) -> list[int]:
    """Pure-function reference: run all *total_layers* of *model_name*.

    Returns the final activation list.  An N-shard
    :class:`LocalPipelineRunner` over the same model must produce an identical
    list -- this is the invariant tests assert.
    """
    return _run_layers(model_name, 0, total_layers, list(values))


class PipelineShardExecutor(TaskExecutor):
    """Execute one model-shard task with the toy deterministic transform.

    Implements the :class:`~computecloud_node.executor.TaskExecutor`
    protocol.  The task payload must carry the shard fields from Phase 9b
    (``start_layer``, ``end_layer``, ``shard_index``, ``shard_count``,
    ``model_name``) plus ``input_activations: list[int]``.

    Returns a dict with ``output_activations``, ``shard_index``, and
    ``is_final`` (``True`` only when ``shard_index == shard_count - 1``).
    """

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if payload is None:
            raise ValueError(
                "PipelineShardExecutor requires a payload with shard fields"
            )
        model_name = payload.get("model_name")
        if not model_name:
            raise ValueError(
                "PipelineShardExecutor payload must contain 'model_name'"
            )
        start_layer = payload.get("start_layer")
        end_layer = payload.get("end_layer")
        if start_layer is None or end_layer is None:
            raise ValueError(
                "PipelineShardExecutor payload must contain 'start_layer' "
                "and 'end_layer'"
            )
        start_layer = int(start_layer)
        end_layer = int(end_layer)
        shard_index = payload.get("shard_index")
        shard_count = payload.get("shard_count")
        if shard_index is None or shard_count is None:
            raise ValueError(
                "PipelineShardExecutor payload must contain 'shard_index' "
                "and 'shard_count'"
            )
        shard_index = int(shard_index)
        shard_count = int(shard_count)

        inputs = payload.get("input_activations")
        if not isinstance(inputs, list):
            raise ValueError(
                "PipelineShardExecutor payload must contain "
                "'input_activations' as a list of integers"
            )
        if not all(isinstance(v, int) for v in inputs):
            raise ValueError(
                "PipelineShardExecutor 'input_activations' must be all ints"
            )
        if start_layer < 0 or end_layer < start_layer:
            raise ValueError(
                f"PipelineShardExecutor invalid layer range "
                f"[{start_layer}, {end_layer})"
            )

        out = _run_layers(str(model_name), start_layer, end_layer, list(inputs))
        return {
            "output_activations": out,
            "shard_index": shard_index,
            "is_final": shard_index == shard_count - 1,
        }


class LocalPipelineRunner:
    """Run an ordered list of shard payloads sequentially, in-process.

    Takes an ordered list of shard task payloads (each carrying ``start_layer``,
    ``end_layer``, ``shard_index``, ``shard_count``, ``model_name``) and an
    initial ``input_activations`` list.  Executes shard 0, feeds its
    ``output_activations`` into shard 1's payload, and so on, returning the
    final shard's output dict.

    Shards run sequentially in the calling thread (no threads, no networking,
    no subprocesses) -- this is the local proof-of-concept for Phase 9c-1.
    """

    def __init__(self, executor: TaskExecutor | None = None) -> None:
        self._executor: TaskExecutor = executor or PipelineShardExecutor()

    def run(
        self,
        shard_payloads: list[dict[str, Any]],
        initial_input: list[int],
    ) -> dict[str, Any]:
        """Execute the shard chain and return the final shard's output.

        Parameters
        ----------
        shard_payloads:
            Ordered list of shard task payloads (shard 0 first).  Each must
            carry the Phase 9b shard fields; ``input_activations`` is injected
            by the runner for every shard (the caller's value on each payload
            is overwritten).
        initial_input:
            The input activation list fed to shard 0.

        Returns
        -------
        The final shard's output dict (``output_activations``,
        ``shard_index``, ``is_final``).
        """
        if not shard_payloads:
            raise ValueError(
                "LocalPipelineRunner requires at least one shard payload"
            )
        # Sanity: shards must be in order 0..N-1.
        for i, p in enumerate(shard_payloads):
            if int(p.get("shard_index", -1)) != i:
                raise ValueError(
                    f"LocalPipelineRunner shards must be ordered 0..N-1; "
                    f"shard at position {i} has shard_index="
                    f"{p.get('shard_index')!r}"
                )
            if int(p.get("shard_count", -1)) != len(shard_payloads):
                raise ValueError(
                    f"LocalPipelineRunner shard_count mismatch: shard {i} "
                    f"reports shard_count={p.get('shard_count')!r} but "
                    f"received {len(shard_payloads)} payloads"
                )

        activations = list(initial_input)
        last_output: dict[str, Any] | None = None
        for i, payload in enumerate(shard_payloads):
            step_payload = dict(payload)
            step_payload["input_activations"] = list(activations)
            out = self._executor.execute(
                task_id=f"shard-{i}",
                job_id=payload.get("job_id", "pipeline-job"),
                payload=step_payload,
            )
            if out is None or "output_activations" not in out:
                raise RuntimeError(
                    f"LocalPipelineRunner shard {i} produced no output"
                )
            activations = list(out["output_activations"])
            last_output = out
        assert last_output is not None
        return last_output


__all__ = [
    "PipelineShardExecutor",
    "LocalPipelineRunner",
    "run_reference",
]


