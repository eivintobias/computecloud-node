"""Tokenizer wrapper for LLM shard executors (Phase 15b, ported to node_client v0.5.0).

Thin wrapper around the ``tokenizers`` library (preferred over the heavier
``transformers`` per the task spec).  The first shard encodes ``prompt ->
token_ids``; the last shard decodes the top-1 logit index -> next token text.

All ``tokenizers`` imports are **lazy** (inside methods) so that importing this
module without tokenizers installed does not raise.
"""

from __future__ import annotations

from typing import Any


class LLMTokenizer:
    """Thin tokenizer wrapper.

    Parameters
    ----------
    tokenizer_path:
        Path to a ``tokenizer.json`` file (from the weights source).
    """

    def __init__(self, tokenizer_path: str) -> None:
        self._path = tokenizer_path
        self._tok: Any = None

    def _ensure_loaded(self) -> None:
        if self._tok is not None:
            return
        from tokenizers import Tokenizer  # lazy import

        self._tok = Tokenizer.from_file(self._path)

    def encode(self, prompt: str) -> list[int]:
        """Encode a text prompt into a list of token IDs."""
        self._ensure_loaded()
        enc = self._tok.encode(prompt)
        return list(enc.ids)

    def decode(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        self._ensure_loaded()
        return self._tok.decode([int(token_id)])

    def vocab_size(self) -> int:
        """Return the tokenizer's vocabulary size."""
        self._ensure_loaded()
        return self._tok.get_vocab_size()


def build_tiny_tokenizer(
    vocab_size: int = 32, sample_text: str = "", save_path: str = ""
) -> str:
    """Build and save a tiny BPE tokenizer for tests.

    Creates a ``tokenizer.json`` at *save_path* and returns the path.
    Uses the ``tokenizers`` library (lazy import).
    """
    from tokenizers import Tokenizer  # lazy import
    from tokenizers.models import BPE  # lazy import
    from tokenizers.trainers import BpeTrainer  # lazy import

    tok = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    text = sample_text or (
        "the quick brown fox jumps over the lazy dog "
        "hello world foo bar baz qux"
    )
    tok.train_from_iterator([text], trainer)
    tok.save(save_path)
    return save_path


__all__ = ["LLMTokenizer", "build_tiny_tokenizer"]
