"""TensorPayload — a stdlib-only tensor wire format (Phase 15a, ported to node_client v0.5.0).

This module defines a **transport-only** description of a tensor that carries no
dependency on any tensor library (no torch, no numpy).  It is the on-the-wire shape
that flows between node shard executors exchanging hidden-state activations for
distributed LLM inference: shard N packs its output hidden states into a
``TensorPayload`` dict (JSON), the server relays it verbatim (it never decodes
tensor bytes), and shard N+1 unpacks it.  Only the endpoints pack/unpack the raw
little-endian payload.

Wire shape (JSON-serialisable dataclass)::

    {
      "shape": [2, 3],
      "dtype": "float32",
      "data_b64": "<base64 of raw little-endian bytes>"
    }

Supported dtypes and their little-endian ``struct`` formats / byte sizes::

    float32  -> 'f'  (4 bytes)
    float16  -> 'e'  (2 bytes)
    int64    -> 'q'  (8 bytes)

Design notes
------------
* **No tensor library.**  Packing/unpacking uses :mod:`struct` only.  ``float16``
  uses the standard ``'e'`` IEEE-754 binary16 format (supported on all CPython
  3.10+ platforms we target).
* **Round-trip guarantees.**  ``float32`` / ``int64`` round-trip **exactly**.
  ``float16`` round-trips within float16 precision (some float32 inputs lose
  precision when narrowed to float16 — that is expected and documented).
* **Size cap.**  A decoded-byte cap (default 256 MB) guards against malicious or
  accidental oversized payloads with a clear :class:`ValueError`.
* **Validation.**  Byte length must equal ``prod(shape) x dtype_size``; unknown
  dtypes are rejected; empty tensors (``prod(shape) == 0``) are allowed with an
  empty byte string.

This is a verbatim port of ``computecloud.tensor_format`` — pure stdlib, no
references to the private ``computecloud`` package.  Kept standalone so the
node_client can exchange activations without importing the server library.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from typing import Any

# ── dtype registry ──────────────────────────────────────────────────────────

# dtype name -> (struct format char, byte size per element)
_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "float32": ("f", 4),
    "float16": ("e", 2),
    "int64": ("q", 8),
}

# Default decoded-byte cap: 256 MB.  Guards against oversized payloads.
DEFAULT_MAX_DECODED_BYTES = 256 * 1024 * 1024


def _prod(shape: tuple[int, ...]) -> int:
    """Product of shape dimensions (1 for the empty shape)."""
    p = 1
    for d in shape:
        p *= d
    return p


def _flatten(nested: list[Any]) -> list[float]:
    """Recursively flatten a nested list into a flat list of numbers."""
    out: list[float] = []
    stack: list[Any] = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        elif isinstance(item, bool):
            out.append(int(item))
        elif isinstance(item, (int, float)):
            out.append(item)
        else:
            raise ValueError(
                f"from_nested_list: non-numeric element {item!r} "
                f"(type {type(item).__name__})"
            )
    return out


def _validate_shape(shape: tuple[int, ...]) -> None:
    """Reject non-tuple shapes, non-int dims, or negative dims."""
    if not isinstance(shape, tuple):
        raise ValueError(f"shape must be a tuple, got {type(shape).__name__}")
    for d in shape:
        if not isinstance(d, int):
            raise ValueError(
                f"shape dimensions must be int, got {type(d).__name__} ({d!r})"
            )
        if d < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {d}")


def _infer_shape(data: Any) -> tuple[int, ...]:
    """Infer the shape of a nested list by walking the first element."""
    shape: list[int] = []
    cur = data
    while isinstance(cur, list):
        shape.append(len(cur))
        if len(cur) == 0:
            break
        cur = cur[0]
    return tuple(shape)


def _reshape(flat: list[Any], shape: tuple[int, ...]) -> list[Any]:
    """Reshape a flat list into a nested list matching *shape* (row-major)."""
    if len(shape) == 0:
        return flat[0] if flat else []  # type: ignore[return-value]
    if len(shape) == 1:
        return list(flat)
    rest = shape[1:]
    chunk_size = _prod(rest)
    if chunk_size == 0:
        return [[] for _ in range(shape[0])]
    return [
        _reshape(flat[i * chunk_size:(i + 1) * chunk_size], rest)
        for i in range(shape[0])
    ]


@dataclass
class TensorPayload:
    """A transport-only tensor description (no tensor library).

    Attributes
    ----------
    shape:
        The tensor shape as a tuple of non-negative ints.
    dtype:
        One of ``"float32"``, ``"float16"``, ``"int64"``.
    data_b64:
        Base64-encoded raw little-endian bytes of the packed tensor data.
        Length must equal ``prod(shape) x dtype_size``.
    """

    shape: tuple[int, ...] = field(default_factory=tuple)
    dtype: str = "float32"
    data_b64: str = ""

    # ── Validation ──────────────────────────────────────────────────────

    def _dtype_info(self) -> tuple[str, int]:
        info = _DTYPE_INFO.get(self.dtype)
        if info is None:
            raise ValueError(
                f"unknown dtype {self.dtype!r}; supported: {sorted(_DTYPE_INFO)}"
            )
        return info

    def validate(self, *, max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES) -> None:
        """Validate shape, dtype, byte length, and size cap.

        Raises ``ValueError`` on any violation.  Empty tensors
        (``prod(shape) == 0``) are allowed with an empty ``data_b64``.
        """
        _validate_shape(self.shape)
        _, elem_size = self._dtype_info()
        n_elems = _prod(self.shape)
        expected_bytes = n_elems * elem_size
        raw = base64.b64decode(self.data_b64) if self.data_b64 else b""
        actual_bytes = len(raw)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"byte length mismatch: data_b64 decodes to {actual_bytes} bytes "
                f"but shape {self.shape} x dtype {self.dtype} (size {elem_size}) "
                f"requires {expected_bytes} bytes"
            )
        if expected_bytes > max_decoded_bytes:
            raise ValueError(
                f"decoded tensor is {expected_bytes} bytes, exceeding the cap "
                f"of {max_decoded_bytes} bytes"
            )

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "data_b64": self.data_b64,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES
    ) -> TensorPayload:
        """Build a :class:`TensorPayload` from a plain dict and validate it."""
        shape_raw = data.get("shape", ())
        if isinstance(shape_raw, list):
            shape = tuple(int(d) for d in shape_raw)
        elif isinstance(shape_raw, tuple):
            shape = tuple(int(d) for d in shape_raw)
        else:
            raise ValueError(
                f"shape must be a list or tuple, got {type(shape_raw).__name__}"
            )
        dtype = str(data.get("dtype", "float32"))
        data_b64 = str(data.get("data_b64", "") or "")
        obj = cls(shape=shape, dtype=dtype, data_b64=data_b64)
        obj.validate(max_decoded_bytes=max_decoded_bytes)
        return obj

    # ── Pack / unpack helpers ───────────────────────────────────────────

    @classmethod
    def from_nested_list(
        cls,
        data: list[Any],
        dtype: str,
        *,
        max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
    ) -> TensorPayload:
        """Build a :class:`TensorPayload` from a nested Python list.

        The list is flattened in row-major (C) order and packed via :mod:`struct`
        into little-endian bytes, then base64-encoded.  The shape is inferred
        from the nesting depth of the first element at each level.

        ``float32`` / ``int64`` round-trip exactly; ``float16`` narrows to IEEE-754
        binary16 (precision loss is expected for values outside the float16 range).
        """
        if dtype not in _DTYPE_INFO:
            raise ValueError(
                f"unknown dtype {dtype!r}; supported: {sorted(_DTYPE_INFO)}"
            )
        shape = _infer_shape(data)
        flat = _flatten(data)
        fmt_char, _ = _DTYPE_INFO[dtype]
        packed = struct.pack(f"<{len(flat)}{fmt_char}", *flat)
        data_b64 = base64.b64encode(packed).decode("ascii")
        obj = cls(shape=shape, dtype=dtype, data_b64=data_b64)
        obj.validate(max_decoded_bytes=max_decoded_bytes)
        return obj

    def to_nested_list(self) -> Any:
        """Unpack the payload into a nested Python list matching ``shape``.

        Returns a scalar when ``shape`` is empty (``()``), and a properly-shaped
        nested list of empty lists when ``prod(shape) == 0`` (e.g. shape
        ``(2, 0)`` -> ``[[], []]``).
        """
        self.validate()
        _, elem_size = self._dtype_info()
        n_elems = _prod(self.shape)
        raw = base64.b64decode(self.data_b64) if self.data_b64 else b""
        fmt_char, _ = _DTYPE_INFO[self.dtype]
        if n_elems == 0:
            if self.shape == ():
                return None
            # Build the properly-shaped empty structure via _reshape with an
            # empty flat list (handles (0,) -> [], (2,0) -> [[], []], etc.).
            return _reshape([], self.shape)
        flat = list(struct.unpack(f"<{n_elems}{fmt_char}", raw))
        if self.shape == ():
            return flat[0]
        return _reshape(flat, self.shape)

    def to_json(self) -> str:
        """Serialize to a JSON string (convenience for transport)."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls, s: str, *, max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES
    ) -> TensorPayload:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s), max_decoded_bytes=max_decoded_bytes)


__all__ = ["TensorPayload", "DEFAULT_MAX_DECODED_BYTES"]
