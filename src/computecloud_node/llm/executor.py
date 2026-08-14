"""LLM shard executor — the TaskExecutor for real LLM layer shards (Phase 15b).

Ported to node_client v0.5.0.  Same logic as ``computecloud.client.llm.executor``
with import paths adapted to ``computecloud_node.*`` and the Phase 15c KV-cache
pass-looping included.  All torch/safetensors/tokenizers imports are **lazy**
(inside methods).

:class:`LLMShardExecutor` implements the
:class:`~computecloud_node.executor.TaskExecutor` protocol.  It recognises
shard payloads whose ``weights_uri`` is ``hf://`` or ``file://``, pulls the
input (a :class:`~computecloud_node.tensor_format.TensorPayload` of hidden states,
or a ``prompt`` string if this is the first shard), runs the shard via
:class:`~computecloud_node.llm.model_shard.TorchShardModule`, and posts the
output (a ``TensorPayload`` of hidden states, or ``{token, token_id}`` if last).

Toy ``model_name``-only shards (no ``weights_uri``) keep routing to the toy
:class:`~computecloud_node.pipeline_executor.PipelineShardExecutor` unchanged.

All torch/safetensors/tokenizers imports are **lazy** (inside methods).
"""

from __future__ import annotations

import base64
import struct
from typing import Any

from computecloud_node.llm.uri import is_llm_weights_uri, parse_file_uri
from computecloud_node.hf_uri import parse_hf_uri
from computecloud_node.tensor_format import TensorPayload

LLM_SHARD_KIND = "llm_shard"


def is_llm_shard_task(payload: dict[str, Any] | None) -> bool:
    """Return True if *payload* is an LLM shard task.

    An LLM shard task has ``kind == "llm_shard"`` AND a ``weights_uri``
    starting with ``hf://`` or ``file://``.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("kind") != LLM_SHARD_KIND:
        return False
    uri = str(payload.get("weights_uri", ""))
    return is_llm_weights_uri(uri)


def _torch_to_tensor_payload(tensor: Any) -> dict[str, Any]:
    """Pack a torch tensor into a TensorPayload dict (float32)."""
    import torch  # lazy

    t = tensor.detach().cpu().float().contiguous()
    shape = tuple(int(d) for d in t.shape)
    flat = t.reshape(-1).tolist()
    n = len(flat)
    packed = struct.pack(f"<{n}f", *flat)
    data_b64 = base64.b64encode(packed).decode("ascii")
    payload = TensorPayload(shape=shape, dtype="float32", data_b64=data_b64)
    payload.validate()
    return payload.to_dict()


def _tensor_payload_to_torch(payload_dict: dict[str, Any], dtype_str: str = "float32") -> Any:
    """Unpack a TensorPayload dict into a torch tensor."""
    import torch  # lazy

    payload = TensorPayload.from_dict(payload_dict)
    nested = payload.to_nested_list()
    dt_map = {"float32": torch.float32, "float16": torch.float16, "int64": torch.long}
    torch_dtype = dt_map.get(payload.dtype, torch.float32)
    return torch.tensor(nested, dtype=torch_dtype)


def probe_llm_capable() -> bool:
    """Return True if the optional [llm] extra (torch) is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False



class LLMShardExecutor:
    """TaskExecutor that runs real LLM layer shards.

    When a task payload is an LLM shard (``is_llm_shard_task`` returns True):
    1. Parses the ``weights_uri`` (``hf://`` via 15a, ``file://`` local).
    2. Creates a ShardWeightsLoader from the source.
    3. Loads only the shard's layer-range weights.
    4. Builds a TorchShardModule, loads weights.
    5. Runs a single forward pass (tokenize if first, argmax if last).
    6. Packs output into TensorPayload (or {token, token_id} if last).

    All heavy imports are lazy.
    """

    def __init__(self, *, cache_dir: str | None = None, max_kv_entries: int = 64) -> None:
        self._cache_dir = cache_dir
        # Phase 15c — per-(run_id, shard_index) KV cache for autoregressive
        # generation.  Each entry holds the list of (k, v) tuples from the last
        # forward pass.  LRU ordering via an OrderedDict-style access list.
        self._kv_cache: dict[tuple[str, int], Any] = {}
        self._kv_order: list[tuple[str, int]] = []
        self._max_kv_entries = max_kv_entries
        self._kv_lock = __import__("threading").Lock()

    def evict_run_id(self, run_id: str) -> int:
        """Drop all KV cache entries for *run_id* (run completed/failed/stopped)."""
        import torch  # lazy — release tensors on GPU/CPU
        with self._kv_lock:
            keys = [k for k in list(self._kv_cache) if k[0] == run_id]
            for k in keys:
                self._kv_cache.pop(k, None)
                if k in self._kv_order:
                    self._kv_order.remove(k)
            return len(keys)

    def _kv_cache_get(self, run_id: str, shard_index: int) -> Any | None:
        """Get (and LRU-bump) the KV cache for (run_id, shard_index)."""
        with self._kv_lock:
            key = (run_id, shard_index)
            if key not in self._kv_cache:
                return None
            if key in self._kv_order:
                self._kv_order.remove(key)
            self._kv_order.append(key)
            return self._kv_cache[key]

    def _kv_cache_put(self, run_id: str, shard_index: int, kv: Any) -> None:
        """Store (and LRU-bump) the KV cache, evicting oldest over the cap."""
        with self._kv_lock:
            key = (run_id, shard_index)
            if key in self._kv_cache:
                if key in self._kv_order:
                    self._kv_order.remove(key)
            self._kv_cache[key] = kv
            self._kv_order.append(key)
            while len(self._kv_cache) > self._max_kv_entries and self._kv_order:
                old = self._kv_order.pop(0)
                self._kv_cache.pop(old, None)

    def kv_cache_size(self) -> int:
        """Number of live KV cache entries (for tests)."""
        with self._kv_lock:
            return len(self._kv_cache)

    def execute(
        self, task_id: str, job_id: str, payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Execute an LLM shard task."""
        if not is_llm_shard_task(payload):
            raise ValueError(
                f"LLMShardExecutor: payload is not an LLM shard task: {task_id}"
            )
        assert payload is not None
        # Phase 15c — cache eviction signal (run finished/failed/stopped).
        evict = payload.get("evict_run_id")
        if evict:
            self.evict_run_id(str(evict))
        return self._run_shard(payload)

    def _run_shard(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the shard forward pass and return the output dict."""
        from computecloud_node.llm.model_shard import TorchShardModule
        from computecloud_node.llm.weights import (
            HFWeightsSource, LocalWeightsSource, ShardWeightsLoader,
        )

        weights_uri = str(payload["weights_uri"])
        config = payload.get("config", {})
        if isinstance(config, str):
            import json
            config = json.loads(config)
        start_layer = int(payload["start_layer"])
        end_layer = int(payload["end_layer"])
        is_first = bool(payload.get("is_first", False))
        is_last = bool(payload.get("is_last", False))
        shard_index = int(payload.get("shard_index", 0))

        # 1. Create the weights source + loader.
        if weights_uri.startswith("file://"):
            dir_path = parse_file_uri(weights_uri)
            source = LocalWeightsSource(dir_path)
        else:
            model_id, revision = parse_hf_uri(weights_uri)
            source = HFWeightsSource(model_id, revision, self._cache_dir)
        loader = ShardWeightsLoader(source)

        # Merge config from the source if available.
        source_config = loader.load_config()
        if source_config:
            for k, v in source_config.items():
                if k not in config:
                    config[k] = v

        # 2. Load only this shard's weights.
        state_dict = loader.load_shard(
            start_layer, end_layer, is_first=is_first, is_last=is_last
        )

        # 3. Build the shard module + load weights.
        module = TorchShardModule(
            config, start_layer, end_layer,
            is_first=is_first, is_last=is_last,
        )
        module.load_state_dict(state_dict)
        return self._prepare_and_run(
            payload, loader, module, config, shard_index, is_first, is_last)

    def _prepare_and_run(self, payload, loader, module, config, shard_index, is_first, is_last):
        """Prepare the input tensor, run the forward pass, pack the output."""
        import torch  # lazy

        use_cache = bool(payload.get("use_cache", False))
        run_id = payload.get("data_run_id")
        pass_index = int(payload.get("pass_index", 0))
        past_kv = None
        if use_cache and run_id is not None:
            past_kv = self._kv_cache_get(str(run_id), shard_index)
        position_offset = int(payload.get("position_offset", 0))
        if position_offset == 0 and past_kv is not None and len(past_kv) > 0:
            try:
                position_offset = int(past_kv[0][0].shape[2])
            except Exception:
                position_offset = 0

        if is_first:
            if use_cache and pass_index > 0:
                ntid = payload.get("next_token_id")
                if ntid is None:
                    raise ValueError(
                        f"LLM shard {shard_index}: generation pass {pass_index} "
                        f"needs 'next_token_id' for shard 0"
                    )
                input_tensor = torch.tensor([int(ntid)], dtype=torch.long)
            else:
                prompt = str(payload.get("prompt", ""))
                tok_path = payload.get("tokenizer_uri")
                if tok_path and tok_path.startswith("file://"):
                    tok_path = parse_file_uri(tok_path)
                elif tok_path is None:
                    tok_path = loader.load_tokenizer_path()
                if tok_path:
                    from computecloud_node.llm.tokenizer import LLMTokenizer
                    tokenizer = LLMTokenizer(tok_path)
                    token_ids = tokenizer.encode(prompt)
                else:
                    token_ids = [ord(c) % config.get("vocab_size", 256) for c in prompt]
                input_tensor = torch.tensor(token_ids, dtype=torch.long)
        else:
            input_payload = payload.get("input")
            if input_payload is None:
                raise ValueError(
                    f"LLM shard {shard_index}: no 'input' TensorPayload"
                )
            input_tensor = _tensor_payload_to_torch(input_payload)

        if use_cache:
            output, new_kv = module.forward(
                input_tensor, past_key_values=past_kv,
                position_offset=position_offset, use_cache=True,
            )
            if run_id is not None:
                self._kv_cache_put(str(run_id), shard_index, new_kv)
        else:
            output = module.forward(input_tensor)
        return self._pack_output(payload, loader, output, shard_index, is_last)

    def _pack_output(self, payload, loader, output, shard_index, is_last):
        """Pack the forward-pass output into the result dict (step 6)."""
        if is_last:
            logits = output  # [batch, seq, vocab]
            last_logits = logits[0, -1, :]  # [vocab]
            token_id = int(last_logits.argmax().item())
            tok_path = payload.get("tokenizer_uri")
            if tok_path and tok_path.startswith("file://"):
                tok_path = parse_file_uri(tok_path)
            elif tok_path is None:
                tok_path = loader.load_tokenizer_path()
            token_text = ""
            if tok_path:
                from computecloud_node.llm.tokenizer import LLMTokenizer
                try:
                    tokenizer = LLMTokenizer(tok_path)
                    token_text = tokenizer.decode(token_id)
                except Exception:
                    token_text = ""
            return {
                "token": token_text, "token_id": token_id,
                "logits": _torch_to_tensor_payload(logits),
                "shard_index": shard_index, "is_final": True,
            }
        return {
            "output": _torch_to_tensor_payload(output),
            "shard_index": shard_index, "is_final": False,
        }

__all__ = [
    "LLMShardExecutor", "is_llm_shard_task", "probe_llm_capable", "LLM_SHARD_KIND",
]
