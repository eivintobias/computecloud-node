"""Minimal ``hf://`` URI parser for the node_client LLM shard executor (Phase 16).

This is a **standalone, minimal** port of the ``parse_hf_uri`` / ``build_hf_uri``
helpers from ``computecloud.scheduler.hf_manifest`` — ONLY the URI-parsing piece,
with no dependency on :class:`computecloud.scheduler.models.ModelManifest` or any
server-side code.  The node_client's :class:`~computecloud_node.llm.executor.LLMShardExecutor`
needs ``parse_hf_uri`` to split an ``hf://{model_id}@{revision}`` weights URI into
its ``(model_id, revision)`` components so :class:`~computecloud_node.llm.weights.HFWeightsSource`
can download the right files.

Pure stdlib — no torch, no numpy, no transformers, no network.

``hf://`` URIs
---------------
Two forms are accepted::

    hf://org/model@revision      # explicit revision
    hf://org/model               # revision defaults to "main"

The ``model_id`` may contain a single ``/`` (org/name) but no leading slashes,
``..`` segments, or whitespace.  Malformed URIs raise :class:`ValueError`.
"""

from __future__ import annotations

import re

# ── HF URI parsing ───────────────────────────────────────────────────────────

_HF_URI_RE = re.compile(
    r"^hf://(?P<model_id>[A-Za-z0-9][A-Za-z0-9._\-/]*?)@(?P<revision>[A-Za-z0-9._\-]+)$"
)
_HF_URI_NO_REV_RE = re.compile(
    r"^hf://(?P<model_id>[A-Za-z0-9][A-Za-z0-9._\-/]*?)$"
)


def parse_hf_uri(uri: str) -> tuple[str, str]:
    """Parse a ``hf://{model_id}@{revision}`` URI.

    Returns ``(model_id, revision)``.  When no ``@revision`` is present, the
    revision defaults to ``"main"``.

    Raises ``ValueError`` for malformed URIs, path traversal (``..``), or
    disallowed characters.  The ``model_id`` may contain a single ``/``
    (org/name) but no leading slashes, ``..`` segments, or whitespace.
    """
    if not isinstance(uri, str):
        raise ValueError(f"hf URI must be a string, got {type(uri).__name__}")
    s = uri.strip()
    if not s.startswith("hf://"):
        raise ValueError(f"hf URI must start with 'hf://', got {uri!r}")
    if ".." in s:
        raise ValueError(f"hf URI must not contain '..': {uri!r}")
    if "\x00" in s or "\n" in s or "\r" in s:
        raise ValueError(f"hf URI must not contain control chars: {uri!r}")
    body = s[len("hf://"):]
    if not body:
        raise ValueError(f"hf URI has empty model_id: {uri!r}")

    m = _HF_URI_RE.match(s)
    if m:
        model_id = m.group("model_id")
        revision = m.group("revision")
    else:
        m2 = _HF_URI_NO_REV_RE.match(s)
        if m2:
            model_id = m2.group("model_id")
            revision = "main"
        else:
            raise ValueError(
                f"malformed hf URI: {uri!r} — expected 'hf://org/name@revision'"
            )

    parts = model_id.split("/")
    if len(parts) > 2:
        raise ValueError(
            f"hf URI model_id must be 'org/name' or 'name', got {model_id!r}"
        )
    for seg in parts:
        if not seg:
            raise ValueError(f"hf URI model_id has empty segment: {model_id!r}")
    if model_id.startswith("/"):
        raise ValueError(f"hf URI model_id must not start with '/': {model_id!r}")
    if "/" in revision:
        raise ValueError(f"hf URI revision must not contain '/': {revision!r}")

    return model_id, revision


def build_hf_uri(model_id: str, revision: str = "main") -> str:
    """Build a ``hf://{model_id}@{revision}`` URI (inverse of parse_hf_uri)."""
    uri = f"hf://{model_id}@{revision}"
    parse_hf_uri(uri)  # raises on bad input
    return uri


__all__ = ["parse_hf_uri", "build_hf_uri"]
