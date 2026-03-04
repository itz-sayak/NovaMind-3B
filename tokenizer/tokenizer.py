"""
Unified tokenizer abstraction.

Supports three backends:
  1. tiktoken cl100k_base  (production baseline, vocab ~100k)
  2. tiktoken gpt2          (legacy, vocab 50k)
  3. Custom SentencePiece   (trained on domain data, vocab 64k)

Usage:
    from tokenizer import get_tokenizer
    tok = get_tokenizer()               # default: cl100k_base
    tok = get_tokenizer("sentencepiece", model_path="/path/to/sp.model")
    tok = get_tokenizer("gpt2")
"""
import os
from typing import Optional, List

import tiktoken


# ─── Vocab-size padding helper ───────────────────────────────────────────────
def _pad_vocab(raw_vocab: int, multiple: int = 64) -> int:
    """Round up to nearest *multiple* so embedding table aligns to GPU tiles."""
    return ((raw_vocab + multiple - 1) // multiple) * multiple


class Tokenizer:
    """Thin wrapper that exposes the same interface regardless of backend."""

    def __init__(
        self,
        backend: str = "cl100k_base",
        model_path: Optional[str] = None,
        vocab_pad_multiple: int = 64,
    ):
        self.backend = backend

        if backend in ("cl100k_base", "gpt2"):
            self._init_tiktoken(backend)
        elif backend == "sentencepiece":
            if model_path is None:
                raise ValueError("model_path required for sentencepiece backend")
            self._init_sentencepiece(model_path)
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")

        self.vocab_size = _pad_vocab(self._raw_vocab, vocab_pad_multiple)

    # ── tiktoken ──────────────────────────────────────────────────────────
    def _init_tiktoken(self, name: str):
        self._enc = tiktoken.get_encoding(name)
        self._raw_vocab = self._enc.n_vocab
        self.eos_token_id = self._enc.eot_token
        self.pad_token_id = self._enc.eot_token
        self.bos_token_id = self._enc.eot_token

    # ── SentencePiece ─────────────────────────────────────────────────────
    def _init_sentencepiece(self, model_path: str):
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._raw_vocab = self._sp.GetPieceSize()
        self.eos_token_id = self._sp.eos_id()
        self.pad_token_id = self._sp.pad_id() if self._sp.pad_id() >= 0 else self.eos_token_id
        self.bos_token_id = self._sp.bos_id() if self._sp.bos_id() >= 0 else self.eos_token_id
        self._enc = None  # sentinel

    # ── Public API ────────────────────────────────────────────────────────
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.encode_ordinary(text)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode without BOS/EOS — used during bulk tokenization."""
        if self.backend == "sentencepiece":
            return self._sp.Encode(text)
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        if self.backend == "sentencepiece":
            # filter padding (IDs >= raw vocab are padding slots)
            tokens = [t for t in tokens if 0 <= t < self._raw_vocab]
            return self._sp.Decode(tokens)
        tokens = [t for t in tokens if t < self._enc.n_vocab]
        return self._enc.decode(tokens)

    def __call__(self, text: str, max_length=None, truncation=True, padding=False):
        tokens = self.encode(text)
        if max_length and len(tokens) > max_length and truncation:
            tokens = tokens[:max_length]
        if padding and max_length and len(tokens) < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        return {"input_ids": tokens}


# ─── Module-level factory ────────────────────────────────────────────────────
_DEFAULT_BACKEND = os.environ.get("TOKENIZER_BACKEND", "cl100k_base")
_DEFAULT_SP_PATH = os.environ.get("TOKENIZER_SP_MODEL", None)


def get_tokenizer(
    backend: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Tokenizer:
    """
    Return a configured Tokenizer.

    Resolution order for `backend`:
      1. Explicit argument
      2. TOKENIZER_BACKEND env-var
      3. "cl100k_base"

    For the sentencepiece backend, `model_path` falls back to the
    TOKENIZER_SP_MODEL env-var.
    """
    backend = backend or _DEFAULT_BACKEND
    if backend == "sentencepiece":
        model_path = model_path or _DEFAULT_SP_PATH
    return Tokenizer(backend=backend, model_path=model_path)
