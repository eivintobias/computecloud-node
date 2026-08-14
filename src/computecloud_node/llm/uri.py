"""URI parsing helpers for LLM shard executors (Phase 15b, ported to node_client v0.5.0).

This module is **pure stdlib** — no torch, no safetensors, no network.  It
complements the :mod:`computecloud_node.hf_uri` ``hf://`` parser
(:func:`computecloud_node.hf_uri.parse_hf_uri`) with a tiny ``file://`` parser so
the executor can recognise local weight directories in tests.

``file://`` URIs
-----------------
Two forms are accepted::

    file:///absolute/path/to/weights_dir
    file:///absolute/path/to/weights_dir/

The path must be absolute (start with ``/`` after the ``file://``) so the
parser is unambiguous across platforms.  Relative paths and ``..`` segments
are rejected.  The returned path is normalised via :func:`os.path.normpath`.

The directory is expected to contain:

* ``model.safetensors`` (single file) **or**
  ``model.safetensors.index.json`` + ``model-*.safetensors`` (sharded),
* ``tokenizer.json`` (optional, for the first shard),
* ``config.json`` (optional, for model config).
"""

from __future__ import annotations

import os
from urllib.parse import unquote, urlparse


def parse_file_uri(uri: str) -> str:
    """Parse a ``file:///path/to/dir`` URI and return the local directory path.

    The path must be absolute.  ``..`` segments and relative paths are rejected
    with :class:`ValueError`.  The returned path is normalised.

    Examples
    --------
    >>> parse_file_uri("file:///tmp/weights")
    '/tmp/weights'
    >>> parse_file_uri("file:///C:/Users/models/llama")
    'C:/Users/models/llama'
    """
    if not isinstance(uri, str):
        raise ValueError(f"file URI must be a string, got {type(uri).__name__}")
    s = uri.strip()
    if not s.startswith("file://"):
        raise ValueError(f"file URI must start with 'file://', got {s!r}")
    parsed = urlparse(s)
    if parsed.scheme != "file":
        raise ValueError(f"file URI scheme must be 'file', got {parsed.scheme!r}")
    # On Windows, file://C:/path produces netloc="C:" — handle this.
    netloc = parsed.netloc
    path = unquote(parsed.path)
    if netloc and netloc not in ("", "localhost"):
        # Check if netloc looks like a Windows drive path (e.g. "C:" or "C:\").
        if len(netloc) >= 2 and netloc[1] == ":" and netloc[0].isalpha():
            path = netloc + path
        else:
            raise ValueError(
                f"file URI with non-local host {netloc!r} is not supported"
            )
    if not path:
        raise ValueError(f"file URI has no path: {s!r}")
    # On Windows, the path may start with a leading slash before the drive
    # letter (e.g. "/C:/Users/...").  Strip exactly one leading slash if the
    # remainder looks like a Windows drive path.
    if (
        len(path) >= 3
        and path[0] == "/"
        and path[2] == ":"
        and path[1].isalpha()
    ):
        path = path[1:]
    if not os.path.isabs(path):
        raise ValueError(
            f"file URI path must be absolute, got {path!r}"
        )
    # Reject path traversal.
    if ".." in path.replace("\\", "/").split("/"):
        raise ValueError(
            f"file URI path must not contain '..' segments, got {path!r}"
        )
    return os.path.normpath(path)


def is_llm_weights_uri(uri: str) -> bool:
    """Return ``True`` if *uri* is an ``hf://`` or ``file://`` weights URI."""
    if not isinstance(uri, str):
        return False
    s = uri.strip()
    return s.startswith("hf://") or s.startswith("file://")


__all__ = ["parse_file_uri", "is_llm_weights_uri"]
