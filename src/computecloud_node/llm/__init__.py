"""Phase 15b — Torch shard executor (optional ``[llm]`` extra), ported to node_client v0.5.0.

This subpackage gives node clients the ability to execute **real** LLM layer
shards: download only their layer range's weights from Hugging Face (or read
local ``.safetensors`` files), run a forward pass over those layers, and
exchange activations via the :class:`~computecloud_node.tensor_format.TensorPayload`
wire format.

**Dependency fence (hard requirement):**

torch / safetensors / tokenizers are allowed **only** inside this subpackage and
**only** as an optional install extra (``pip install -e ".[llm]"``).  Every
heavy import is done **lazily** — inside the function/method that needs it —
so that ``import computecloud_node`` (which does NOT import this subpackage)
keeps working without torch installed.  No module outside ``llm/`` may
import torch, numpy, safetensors, or transformers, not even under
``TYPE_CHECKING``.

Public API (re-exported here for convenience; still requires the ``[llm]``
extra to actually *use*)::

    from computecloud_node.llm import LLMShardExecutor, TorchShardModule
"""

from __future__ import annotations

# Nothing heavy is imported at package level.  The classes below are available
# as attribute lookups (``computecloud_node.llm.LLMShardExecutor``) but are
# only materialised when the caller actually accesses them, at which point the
# lazy import inside the class/function fires.

__all__ = [
    "LLMShardExecutor",
    "TorchShardModule",
    "ShardWeightsLoader",
    "LocalWeightsSource",
    "HFWeightsSource",
    "LLMTokenizer",
    "is_llm_shard_task",
    "probe_llm_capable",
]


def __getattr__(name: str):
    """Lazily resolve the re-exported symbols on first attribute access.

    Nothing heavy is imported at package import time; the classes/functions are
    only materialised when a caller actually accesses them (e.g.
    ``computecloud_node.llm.LLMShardExecutor``), at which point the lazy import
    inside the target module fires.
    """
    if name in __all__:
        if name in ("LLMShardExecutor", "is_llm_shard_task", "probe_llm_capable"):
            from computecloud_node.llm import executor as _m
            return getattr(_m, name)
        if name == "TorchShardModule":
            from computecloud_node.llm import model_shard as _m
            return getattr(_m, name)
        if name in ("ShardWeightsLoader", "LocalWeightsSource", "HFWeightsSource"):
            from computecloud_node.llm import weights as _m
            return getattr(_m, name)
        if name == "LLMTokenizer":
            from computecloud_node.llm import tokenizer as _m
            return getattr(_m, name)
    raise AttributeError(f"module 'computecloud_node.llm' has no attribute {name!r}")

