"""Shard weights loader — partial safetensors loading per layer range (Phase 15b).

Ported to node_client v0.5.0.  Same logic as ``computecloud.client.llm.weights``,
with all heavy (safetensors/httpx) imports kept lazy inside methods so this
module imports cleanly without the ``[llm]`` extra.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Protocol

_LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")


def keys_for_shard(all_keys, start_layer, end_layer, *, is_first, is_last):
    """Filter all_keys to keys belonging to layers [start, end)."""
    wanted = []
    idxs = set(range(start_layer, end_layer))
    for key in all_keys:
        m = _LAYER_KEY_RE.match(key)
        if m:
            if int(m.group(1)) in idxs:
                wanted.append(key)
            continue
        if is_first and key == "model.embed_tokens.weight":
            wanted.append(key)
        if is_last and key in ("model.norm.weight", "lm_head.weight", "model.lm_head.weight"):
            wanted.append(key)
    return wanted


def _shard_sort_key(fn):
    m = re.search(r"(\d+)-of-(\d+)", fn)
    return (int(m.group(1)), fn) if m else (0, fn)


class WeightsSource(Protocol):
    def list_keys(self) -> list[str]: ...
    def get_tensor(self, key: str) -> Any: ...
    def get_config(self) -> dict[str, Any]: ...
    def get_tokenizer_path(self) -> str | None: ...


class LocalWeightsSource:
    """Read weights from a local directory of .safetensors files."""

    def __init__(self, dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"weights directory not found: {dir_path}")
        self._dir = dir_path
        self._index = None
        self._config = None
        self._all_keys = None
        self._cache = {}

    def _safe_open(self, filename):
        if filename not in self._cache:
            from safetensors import safe_open
            path = os.path.join(self._dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"safetensors file not found: {path}")
            self._cache[filename] = safe_open(path, framework="pt")
        return self._cache[filename]

    def _ensure_index(self):
        if self._index is not None:
            return
        single = os.path.join(self._dir, "model.safetensors")
        idx = os.path.join(self._dir, "model.safetensors.index.json")
        if os.path.isfile(single):
            f = self._safe_open("model.safetensors")
            self._index = {k: "model.safetensors" for k in f.keys()}
        elif os.path.isfile(idx):
            with open(idx, encoding="utf-8") as fh:
                self._index = dict(json.load(fh).get("weight_map", {}))
        else:
            raise FileNotFoundError(f"no safetensors in {self._dir}")

    def list_keys(self):
        self._ensure_index()
        if self._all_keys is None:
            self._all_keys = list(self._index.keys())
        return list(self._all_keys)

    def get_tensor(self, key):
        self._ensure_index()
        fn = self._index.get(key)
        if fn is None:
            raise KeyError(f"key not found: {key}")
        return self._safe_open(fn).get_tensor(key)

    def get_config(self):
        if self._config is None:
            p = os.path.join(self._dir, "config.json")
            self._config = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
        return dict(self._config)

    def get_tokenizer_path(self):
        p = os.path.join(self._dir, "tokenizer.json")
        return p if os.path.isfile(p) else None


class HFWeightsSource:
    """Download weights from Hugging Face via the resolve URLs.

    Caches downloaded files under ``cache_dir`` (default: a temp dir) and tracks
    total cache size against ``max_cache_bytes``.  All network I/O is lazy
    (httpx imported inside the download method) so this module imports cleanly
    without httpx installed.
    """

    def __init__(self, model_id, revision="main", cache_dir=None,
                 max_cache_bytes=8 * 1024 * 1024 * 1024):
        self.model_id = model_id
        self.revision = revision
        self.max_cache = max_cache_bytes
        if cache_dir:
            self._cache_root = cache_dir
            os.makedirs(self._cache_root, exist_ok=True)
        else:
            self._cache_root = tempfile.mkdtemp(prefix="cc_llm_hf_")
        self._cache_size = 0
        self._cache = {}
        self._index = None
        self._config = None

    def _cache_path(self, filename):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
        return os.path.join(self._cache_root, f"{self.model_id.replace('/', '_')}_{safe}")

    def index_url(self):
        return f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/model.safetensors.index.json"

    def shard_url(self, filename):
        return f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/{filename}"

    def config_url(self):
        return f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/config.json"

    def tokenizer_url(self):
        return f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/tokenizer.json"

    def _cache_has(self, filename):
        return os.path.isfile(self._cache_path(filename))

    @property
    def cache_size_bytes(self):
        return self._cache_size

    def _download(self, url, dest):
        import httpx
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
        self._cache_size += os.path.getsize(dest)

    def _safe_open(self, filename):
        if filename not in self._cache:
            from safetensors import safe_open
            cp = self._cache_path(filename)
            if not self._cache_has(filename):
                self._download(self.shard_url(filename), cp)
            self._cache[filename] = safe_open(cp, framework="pt")
        return self._cache[filename]

    def _ensure_index(self):
        if self._index is not None:
            return
        ip = self._cache_path("model.safetensors.index.json")
        if not self._cache_has("model.safetensors.index.json"):
            try:
                self._download(self.index_url(), ip)
            except Exception:
                self._index = {}
                f = self._safe_open("model.safetensors")
                self._index = {k: "model.safetensors" for k in f.keys()}
                return
        with open(ip, encoding="utf-8") as fh:
            self._index = dict(json.load(fh).get("weight_map", {}))

    def list_keys(self):
        self._ensure_index()
        return list(self._index.keys())

    def get_tensor(self, key):
        self._ensure_index()
        fn = self._index.get(key)
        if fn is None:
            raise KeyError(f"key not found: {key}")
        return self._safe_open(fn).get_tensor(key)

    def get_config(self):
        if self._config is None:
            cp = self._cache_path("config.json")
            if not self._cache_has("config.json"):
                self._download(self.config_url(), cp)
            with open(cp, encoding="utf-8") as fh:
                self._config = json.load(fh)
        return dict(self._config)

    def get_tokenizer_path(self):
        tp = self._cache_path("tokenizer.json")
        if not self._cache_has("tokenizer.json"):
            try:
                self._download(self.tokenizer_url(), tp)
            except Exception:
                return None
        return tp


class ShardWeightsLoader:
    """Load only the weights for a layer-shard range from a weights source."""

    def __init__(self, source):
        self._source = source

    def load_shard(self, start_layer, end_layer, *, is_first, is_last):
        """Load state dict for layers [start, end) + overheads."""
        all_keys = self._source.list_keys()
        wanted = keys_for_shard(
            all_keys, start_layer, end_layer, is_first=is_first, is_last=is_last)
        return {k: self._source.get_tensor(k) for k in wanted}

    def load_config(self):
        return self._source.get_config()

    def load_tokenizer_path(self):
        return self._source.get_tokenizer_path()

    @property
    def source(self):
        return self._source


__all__ = [
    "ShardWeightsLoader", "LocalWeightsSource", "HFWeightsSource",
    "WeightsSource", "keys_for_shard",
]
