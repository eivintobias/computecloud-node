"""ComputeCloud Node Client — share your compute with the pool.

Public API:
    NodeConfig        Configuration for connecting to the pool coordinator.
    NodeCapabilities  Static hardware advertisement.
    TaskExecutor      Protocol implementable by users to define task logic.
    TaskResult        Outcome of a task execution.
    LocalProcessExecutor  Built-in executor that runs shell commands.
    DockerExecutor    Built-in executor that runs tasks in Docker containers.
    ComputeNode       Main entry point — registers, polls, executes, reports.
    PipelineShardExecutor  Toy shard executor (deterministic, no ML libs).
    PipelineShardWorker    Node-side worker for distributed pipeline shards.
    ShardAwareExecutor     Adapter that routes shard tasks to the worker.
    DataShardWorker        Node-side worker for generalized data-run shards.
    DataShardAwareExecutor Universal router: data shards, pipeline shards, plain tasks.

Quick start::

    git clone https://github.com/eivintobias/computecloud-node.git
    cd computecloud-node
    pip install -e .

    python -m computecloud_node --http \\
        --api-url https://api.computepool.cloud --executor local \\
        --cpu 4 --ram 16 --username admin --password YOUR_PASSWORD

HTTP mode requires only httpx (installed by default).
gRPC mode (optional) requires the grpc extra::

    pip install -e ".[grpc]"
"""

from __future__ import annotations

from computecloud_node.config import (
    NodeCapabilities,
    NodeConfig,
)
from computecloud_node.data_worker import (
    DATA_MERGE_KIND,
    DATA_SHARD_KIND,
    DataShardAwareExecutor,
    DataShardWorker,
    is_data_shard_task,
)
from computecloud_node.docker_executor import DockerExecutor
from computecloud_node.executor import (
    TaskExecutor,
    TaskResult,
)
from computecloud_node.local_executor import LocalProcessExecutor
from computecloud_node.node import ComputeNode
from computecloud_node.pipeline_executor import (
    LocalPipelineRunner,
    PipelineShardExecutor,
    run_reference,
)
from computecloud_node.pipeline_worker import PipelineShardWorker, is_shard_task
from computecloud_node.shard_executor_adapter import ShardAwareExecutor


def __getattr__(name: str):
    """Lazy attribute access for the optional ``[llm]`` extra symbols.

    These live in :mod:`computecloud_node.llm` (torch/safetensors/tokenizers),
    which is **not** imported at package level so that ``import
    computecloud_node`` keeps working without torch installed.  The symbols are
    only materialised when a caller actually accesses them — at which point the
    lazy imports inside ``computecloud_node.llm.*`` fire (and raise
    ``ImportError`` if the ``[llm]`` extra is absent, never silently).
    """
    _LLM_EXPORTS = {
        "LLMShardExecutor",
        "TorchShardModule",
        "ShardWeightsLoader",
        "LocalWeightsSource",
        "HFWeightsSource",
        "LLMTokenizer",
        "is_llm_shard_task",
        "probe_llm_capable",
    }
    if name in _LLM_EXPORTS:
        from computecloud_node import llm as _llm  # noqa: PLC0415

        return getattr(_llm, name)
    raise AttributeError(f"module 'computecloud_node' has no attribute {name!r}")


__all__ = [
    "NodeConfig",
    "NodeCapabilities",
    "TaskExecutor",
    "TaskResult",
    "LocalProcessExecutor",
    "DockerExecutor",
    "ComputeNode",
    # Pipeline executor -- local PoC (Phase 9c-1)
    "PipelineShardExecutor",
    "LocalPipelineRunner",
    "run_reference",
    # Distributed shard worker -- Phase 9c-2
    "PipelineShardWorker",
    "ShardAwareExecutor",
    "is_shard_task",
    # Generalized data shard worker -- Phase 13c (ported v0.4.0)
    "DataShardWorker",
    "DataShardAwareExecutor",
    "is_data_shard_task",
    "DATA_SHARD_KIND",
    "DATA_MERGE_KIND",
    # LLM shard executor -- Phase 16 (optional [llm] extra, lazy)
    "LLMShardExecutor",
    "TorchShardModule",
    "ShardWeightsLoader",
    "LocalWeightsSource",
    "HFWeightsSource",
    "LLMTokenizer",
    "is_llm_shard_task",
    "probe_llm_capable",
]

