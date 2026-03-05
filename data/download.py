"""
Data downloading for NovaMind-3B training.

Pretraining data target: ~350GB on disk → ~120B tokens

  Existing (phase-1 base):
  - OpenWebText:       8M docs,  ~38GB  (general web)
  - TheStack Python:   5M files, ~80GB  (code)
  - OpenWebMath:       6.3M,     ~30GB  (math)
  - MetaMathQA:        395K,     ~350MB (math QA)
  - FineWeb-Edu:       2M docs,  ~8GB   (educational)
  - Wikipedia EN:      6.4M,     ~20GB  (encyclopaedic)
  - C4 English:        3M docs,  ~12GB  (general web)
  - CC-News:           1M docs,  ~4GB   (news)
  - TheStack Java+JS:  2M each,  ~20GB  (code)

  New (download_new_pretrain_data):
  - CodeSearchNet:        2M,  ~4GB  (structured docstring+code, py/java/js)
  - GitHub Code:          1M,  ~8GB  (codeparrot/github-code, py/java/js, filtered)
  - RedPajama arXiv:      1M,  ~15GB (28B-token arXiv slice, math/CS/physics)
  - RedPajama SE:         500K,~3GB  (stackexchange slice as long-form text)
  - lvwerra SE paired:    500K,~5GB  (StackExchange Q&A, Markdown, 26.8M total)
  - FLAN collection:      500K,~2GB  (diverse NLP tasks + templates, instruction format)
  - SmolTalk:             300K,~3GB  (multi-turn conversational instruction pairs)
  - Alpaca:                52K,~50MB (GPT-3.5 general instruction following)

SFT  (~5GB)
DPO  (~1GB)

Quality pipeline applied to new sources:
  - Text quality gate (length, symbol ratio)
  - English language detection (function-word heuristic)
  - Python AST parseability check for Python code
  - Rolling MD5-hash deduplication (intra-source)
  - Metadata provenance tag (__source__) in every row
"""
import os
import sys
import builtins

# Patch print() to always flush — must be before any other import or print.
# conda run wraps stdout in a pipe, making Python block-buffer by default.
# This ensures every message appears immediately regardless of the wrapper.
_real_print = builtins.print
def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault("flush", True)
    _real_print(*args, **kwargs)
builtins.print = print

import json
import sys
import time
import argparse
import itertools
import shutil
import hashlib
import ast as _ast
import re as _re
import numpy as np

# ── Redirect ALL HuggingFace I/O to /mnt/zone/A BEFORE any HF import ──────────
# `cache_dir` in load_dataset() only controls processed Arrow output.
# Raw parquet blob downloads go to HF_HOME/hub/ — must override here or
# they land in ~/.cache and fill the home partition.
_HF_HOME = os.environ.get("HF_HOME", "/mnt/zone/A/datasets/hf_cache")
os.environ["HF_HOME"]          = _HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_HOME, "datasets")
os.environ["HF_HUB_CACHE"]     = os.path.join(_HF_HOME, "hub")
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from huggingface_hub import login

# ── Chunked-write settings ──────────────────────────────────
CHUNK_ROWS = 50_000  # rows per Arrow shard — keeps RSS ~1-2 GB
_FORCE     = False   # set to True via --force to re-download existing sources

# ── Load HF token ────────────────────────────────────────────────────────────
# Search for .env in: script dir, project root, home dir, cwd (in that order)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_SEARCH = [
    os.path.join(_PROJECT_ROOT, ".env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.path.expanduser("~"), ".env"),
    os.path.join(os.getcwd(), ".env"),
]


def _load_env_file(path: str) -> bool:
    """
    Parse a .env file and inject variables into os.environ.

    Handles:
      - KEY=VALUE
      - export KEY=VALUE          (shell-style)
      - KEY="VALUE WITH SPACES"   (double-quoted values)
      - KEY='VALUE'               (single-quoted values)
      - # comment lines
      - Blank lines
      - Values that themselves contain '='
    """
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Strip leading 'export ' (case-insensitive)
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip surrounding quotes (" or ')
            if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                val = val[1:-1]
            # Only set if not already present (env > .env)
            if key and key not in os.environ:
                os.environ[key] = val
    return True


_env_loaded = False
for _env_path in _ENV_SEARCH:
    if _load_env_file(_env_path):
        print(f"✓ Loaded .env from {_env_path}")
        _env_loaded = True
        break

if not _env_loaded:
    print(f"⚠ No .env file found (searched: {', '.join(_ENV_SEARCH)})")

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)
    print(f"✓ Logged in to HuggingFace Hub (token: ...{_hf_token[-4:]})")
else:
    print("⚠ No HF_TOKEN found. Gated datasets may fail. Set HF_TOKEN in .env or environment.")

BASE_DIR = "/mnt/zone/A/datasets"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Quality filtering, language detection, and deduplication utilities
# ─────────────────────────────────────────────────────────────────────────────

# Top-200 English function words for fast in-process language detection
_ENGLISH_FN_WORDS = frozenset(
    "the of and to a in is it you that he was for on are with as at be by "
    "have this from or an they we say her she will my one all would there "
    "their what so up out if about who get which go me when make can like time "
    "no just him know take people into year your good some could them see other "
    "than then now look only come its over think also back after use two how our "
    "work first well way even new want because any these give day most us".split()
)


def is_english(text: str, sample_words: int = 80, threshold: float = 0.55) -> bool:
    """Fast English detection via common function-word ratio (no external deps)."""
    words = [w.strip(".,!?;:\"'()[]{}\u2014") for w in text.lower().split()[:sample_words]]
    if not words:
        return False
    en = sum(1 for w in words if w in _ENGLISH_FN_WORDS)
    return en / len(words) >= threshold


def text_quality(text: str, min_len: int = 200, max_symbol_ratio: float = 0.25) -> bool:
    """Baseline quality gate: minimum length and symbol density."""
    if len(text) < min_len:
        return False
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / max(len(text), 1) <= max_symbol_ratio


def math_quality(text: str) -> bool:
    """Accept text with reasonable equation/math-symbol density or sufficient length."""
    if len(text) < 100:
        return False
    math_tokens = (
        text.count('$') + text.count('\\') +
        text.count('\u222b') + text.count('\u2211') +  # ∫ ∑
        text.count('\u2264') + text.count('\u2265') + text.count('\u2208')  # ≤ ≥ ∈
    )
    density = math_tokens / max(len(text), 1)
    return density >= 0.002 or len(text) > 800


def code_quality_python(code: str, min_lines: int = 3, max_lines: int = 500) -> bool:
    """Python code quality: parseable AST, has functions/classes, sane length."""
    lines = code.split('\n')
    if not (min_lines <= len(lines) <= max_lines):
        return False
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
        for n in _ast.walk(tree)
    )


def code_quality_generic(code: str, min_lines: int = 3, max_lines: int = 600) -> bool:
    """Length-only quality gate for non-Python code."""
    lines = code.split('\n')
    return min_lines <= len(lines) <= max_lines


_HTML_TAG_RE = _re.compile(r'<[^>]+')


def strip_html(text: str) -> str:
    """Remove HTML tags (StackExchange answers contain HTML)."""
    return _HTML_TAG_RE.sub(' ', text).strip()


# ── Rolling-hash deduplication (in-process, per-source) ──────────────────────
_SEEN_HASHES: set[int] = set()


def is_duplicate(text: str, window_chars: int = 256) -> bool:
    """MD5-fingerprint dedup on first `window_chars` chars of normalised text."""
    fp = ' '.join(text[:window_chars].lower().split())
    h = int(hashlib.md5(fp.encode('utf-8', errors='ignore')).hexdigest(), 16) & 0xFFFF_FFFF
    if h in _SEEN_HASHES:
        return True
    _SEEN_HASHES.add(h)
    return False


def reset_dedup():
    """Clear the dedup fingerprint set between sources."""
    global _SEEN_HASHES
    _SEEN_HASHES = set()
    print('  \u2192 Dedup fingerprint set cleared')


def stream_and_save(dataset_name, hf_name, out_path, n_samples, desc=None,
                    hf_kwargs=None, chunk_rows=CHUNK_ROWS,
                    filter_fn=None, transform_fn=None, meta_fields=None):
    """
    Stream n_samples **passing** ``filter_fn`` from a HF dataset in chunks.

    Peak RSS stays ~1-2 GB regardless of total dataset size.

    Args:
        filter_fn:    callable(item) -> bool  — skip items returning False.
        transform_fn: callable(item) -> item | None  — reshape item before
                      saving; return None to discard the item.
        meta_fields:  dict merged into every saved item (provenance tags).
    """
    if os.path.exists(out_path):
        if not _FORCE:
            print(f"  ✓ {dataset_name} already exists at {out_path}, skipping.")
            return True
        print(f"  ⚠ --force: removing existing {out_path}")
        shutil.rmtree(out_path)

    # Remove any stale shard directory from a previous interrupted run.
    shard_dir_early = out_path + "__shards"
    if os.path.exists(shard_dir_early):
        shutil.rmtree(shard_dir_early)

    if hf_kwargs is None:
        hf_kwargs = {}

    label = desc or dataset_name
    filter_tag = " +filter" if filter_fn else ""
    print(f"\n  ↓ Downloading {label} ({n_samples:,} kept{filter_tag}, chunk={chunk_rows:,})...")
    try:
        ds = load_dataset(hf_name, streaming=True, token=_hf_token,
                          cache_dir=CACHE_DIR, **hf_kwargs)
    except Exception as e:
        print(f"  ✗ Failed to load {hf_name}: {e}")
        return False

    # Temporary directory for Arrow shards
    shard_dir = out_path + "__shards"
    os.makedirs(shard_dir, exist_ok=True)

    chunk_buf = []
    shard_paths = []
    total_collected = 0
    total_seen = 0

    pbar = tqdm(
        total=n_samples, desc=f"  {dataset_name}", unit=" docs", ncols=110,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    def _flush_chunk(buf, shard_idx):
        """Write a chunk of rows to an Arrow shard and free memory."""
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
        ds_chunk = Dataset.from_list(buf)
        ds_chunk.save_to_disk(shard_path)
        shard_paths.append(shard_path)
        del ds_chunk
        buf.clear()

    try:
        for item in ds:
            total_seen += 1

            # ── Optional quality / dedup filter ──────────────────────────────
            if filter_fn is not None and not filter_fn(item):
                continue

            # ── Optional reshape / text-field normalisation ───────────────────
            if transform_fn is not None:
                item = transform_fn(item)
                if item is None:
                    continue

            # ── Provenance metadata ───────────────────────────────────────────
            if meta_fields:
                item = {**item, **meta_fields}

            chunk_buf.append(item)
            total_collected += 1
            pbar.update(1)

            if len(chunk_buf) >= chunk_rows:
                _flush_chunk(chunk_buf, len(shard_paths))

            if total_collected >= n_samples:
                break
    except Exception as e:
        pbar.close()
        print(f"  ✗ Error during streaming: {e}")
        if not chunk_buf and not shard_paths:
            shutil.rmtree(shard_dir, ignore_errors=True)
            return False
        print(f"  → Saving partial: {total_collected:,} samples collected")

    pbar.close()
    if filter_fn and total_seen > 0:
        keep_rate = total_collected / total_seen * 100
        print(f"  → Filter keep-rate: {total_collected:,}/{total_seen:,} = {keep_rate:.1f}%")

    # Flush remaining rows
    if chunk_buf:
        _flush_chunk(chunk_buf, len(shard_paths))

    if not shard_paths:
        print(f"  ✗ No data collected for {dataset_name}")
        shutil.rmtree(shard_dir, ignore_errors=True)
        return False

    # Concatenate shards into final dataset
    print(f"  → Merging {len(shard_paths)} shards into {out_path}...")
    shards = [load_from_disk(p) for p in shard_paths]
    full_ds = concatenate_datasets(shards)
    full_ds.save_to_disk(out_path)

    size_gb = sum(
        os.path.getsize(os.path.join(out_path, f))
        for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))
    ) / (1024**3)
    print(f"  ✓ Saved {len(full_ds):,} examples to {out_path} ({size_gb:.1f} GB)")

    # Clean up shards
    del shards, full_ds
    shutil.rmtree(shard_dir, ignore_errors=True)
    return True


def download_full(hf_name, out_path, dataset_name, desc=None, hf_kwargs=None):
    """Download a full (non-streaming) dataset with HF built-in progress bars."""
    if os.path.exists(out_path):
        print(f"  ✓ {dataset_name} already exists at {out_path}, skipping.")
        return True

    if hf_kwargs is None:
        hf_kwargs = {}

    print(f"\n  ↓ Downloading {desc or dataset_name} (full)...")
    try:
        ds = load_dataset(hf_name, token=_hf_token, cache_dir=CACHE_DIR, **hf_kwargs)
        ds.save_to_disk(out_path)
        n = len(ds) if not isinstance(ds, dict) else sum(len(v) for v in ds.values())
        print(f"  ✓ Saved {n:,} examples to {out_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming tokeniser — write uint32 tokens directly to a .bin without saving
# intermediate Arrow files.  Compatible with train.bin / val.bin format.
# ─────────────────────────────────────────────────────────────────────────────

def stream_tokenize_to_bin(
    source_name: str,
    hf_name: str,
    text_field: str,
    bin_path: str,
    n_tokens_target: int,
    hf_kwargs: dict | None = None,
    filter_fn=None,
    min_len: int = 200,
) -> int:
    """Stream a HF dataset, tokenize, and append uint32 tokens directly to bin_path.

    No intermediate Arrow files are created.  Resumable via a companion
    ``<bin_path>.<source_name>_progress`` JSON file that records:
      - offset_bytes  : byte position in bin_path at which this source started
      - docs_seen     : HF rows consumed (for fast-skip on resume)
      - tokens_written: tokens appended in this run
      - done          : True once the source hit n_tokens_target

    On restart the bin file is truncated back to offset_bytes and the first
    docs_seen rows are fast-skipped (iterated but not tokenised).

    Args:
        source_name:     Unique identifier used for the progress file.
        hf_name:         HuggingFace dataset identifier.
        text_field:      Row key containing the document text.
        bin_path:        Absolute path to the target .bin file (e.g. train.bin).
        n_tokens_target: Stop after this many tokens have been appended.
        hf_kwargs:       Extra kwargs forwarded to load_dataset (split, name …).
        filter_fn:       Optional callable(item) → bool; rows returning False
                         are skipped without tokenising.
        min_len:         Minimum character length of text to tokenise.

    Returns:
        Number of tokens appended in this invocation (0 on load failure).
    """
    # ── Lazy import of project tokenizer ────────────────────────────────────
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from tokenizer.tokenizer import get_tokenizer  # noqa: PLC0415

    progress_file = bin_path + f".{source_name}_progress"

    # ── Load / initialise progress state ────────────────────────────────────
    state: dict = {"offset_bytes": 0, "docs_seen": 0, "tokens_written": 0, "done": False}
    if os.path.exists(progress_file):
        with open(progress_file) as _pf:
            state = json.load(_pf)

    if state.get("done"):
        tok_b = state["tokens_written"] / 1e9
        print(f"  ✓ {source_name} already complete ({tok_b:.0f}B tokens), skipping.")
        return state["tokens_written"]

    # ── Verify / repair bin file alignment with recorded offset ─────────────
    current_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
    saved_offset = state["offset_bytes"]

    if current_size < saved_offset:
        # Bin was truncated externally — can't trust offset, restart
        print(f"  ⚠ {source_name}: bin size {current_size:,} < saved offset {saved_offset:,}")
        print(f"  → Restarting {source_name} from scratch (docs/offset reset).")
        state = {"offset_bytes": current_size, "docs_seen": 0, "tokens_written": 0, "done": False}
    elif current_size > saved_offset:
        if state["docs_seen"] > 0 or state["tokens_written"] > 0:
            # Crashed mid-write — this source already started writing, truncate back
            print(f"  ⚠ {source_name}: bin {current_size:,} > offset {saved_offset:,}, rolling back.")
            with open(bin_path, "r+b") as _bf:
                _bf.truncate(saved_offset)
            state["docs_seen"] = 0
            state["tokens_written"] = 0
            print(f"  → Restarting {source_name} (rolled back to offset {saved_offset:,}).")
        # else: fresh start — bin already has data from prior stages; the anchor
        # below will set offset_bytes = current_size so we safely append after it.
    # else current_size == saved_offset → resume normally

    # If this is a fresh start, anchor the offset to current bin EOF (safe append point)
    if state["docs_seen"] == 0 and state["tokens_written"] == 0:
        state["offset_bytes"] = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0

    tok_done = state["tokens_written"] / 1e9
    tok_tgt = n_tokens_target / 1e9
    print(f"\n  ↓ Streaming {source_name} → {bin_path}")
    print(f"    Target: {tok_tgt:.0f}B tokens | Resumed: {tok_done:.1f}B done | Skip {state['docs_seen']:,} docs")

    # ── Load streaming dataset ───────────────────────────────────────────────
    if hf_kwargs is None:
        hf_kwargs = {}
    try:
        ds = load_dataset(hf_name, streaming=True, token=_hf_token, **hf_kwargs)
    except Exception as exc:
        print(f"  ✗ Failed to load {hf_name}: {exc}")
        return 0

    tok_obj = get_tokenizer()
    eos_id  = tok_obj.eos_token_id

    buf:      list[int] = []
    BUF_TOKS  = 500_000       # flush to disk every ~2 MB
    CKPT_TOKS = 50_000_000    # write progress file every 50 M tokens

    docs_seen      = state["docs_seen"]
    tokens_written = state["tokens_written"]
    skip_count     = docs_seen     # fast-skip rows already processed in a prior run
    last_ckpt      = tokens_written

    pbar = tqdm(
        total=n_tokens_target,
        initial=tokens_written,
        desc=f"  {source_name}",
        unit=" tok",
        ncols=110,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    def _flush(f_out):
        if buf:
            np.array(buf, dtype=np.uint32).tofile(f_out)
            buf.clear()

    def _save_progress(done: bool = False) -> None:
        cur = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
        with open(progress_file, "w") as _pf:
            json.dump({
                "offset_bytes":   state["offset_bytes"],
                "docs_seen":      docs_seen,
                "tokens_written": tokens_written,
                "done":           done,
                "bin_size":       cur,
            }, _pf)

    try:
        with open(bin_path, "ab") as f_out:
            for item in ds:
                # Fast-skip rows already processed on a previous run
                if skip_count > 0:
                    skip_count -= 1
                    continue

                docs_seen += 1

                # Optional quality / dedup filter
                if filter_fn is not None and not filter_fn(item):
                    continue

                text = item.get(text_field, "")
                if len(text) < min_len:
                    continue

                toks = tok_obj.encode_ordinary(text)
                if not toks:
                    continue
                toks.append(eos_id)

                buf.extend(toks)
                tokens_written += len(toks)
                pbar.update(len(toks))

                if len(buf) >= BUF_TOKS:
                    _flush(f_out)

                if tokens_written - last_ckpt >= CKPT_TOKS:
                    _flush(f_out)
                    f_out.flush()
                    _save_progress()
                    last_ckpt = tokens_written
                    gb_now = os.path.getsize(bin_path) / (1024 ** 3)
                    import shutil as _shutil
                    free_gb = _shutil.disk_usage(os.path.dirname(bin_path)).free / (1024 ** 3)
                    print(f"\n    [{source_name}] checkpoint: {tokens_written/1e9:.1f}B tok | train.bin {gb_now:.1f} GB | disk free {free_gb:.0f} GB")
                    if free_gb < 50:
                        print(f"  ⚠  Disk nearly full ({free_gb:.0f} GB free) — stopping {source_name}. Resume later.")
                        break

                if tokens_written >= n_tokens_target:
                    break

            _flush(f_out)

    except KeyboardInterrupt:
        pbar.close()
        _save_progress()
        print(f"\n  ⏸ {source_name} interrupted at {tokens_written/1e9:.1f}B tokens — resumable.")
        return tokens_written
    except Exception as exc:
        pbar.close()
        _save_progress()
        print(f"\n  ✗ {source_name} error: {exc}")
        import traceback; traceback.print_exc()
        return tokens_written

    pbar.close()
    _save_progress(done=True)

    size_gb = os.path.getsize(bin_path) / (1024 ** 3)
    print(f"  ✓ {source_name}: {tokens_written/1e9:.1f}B tokens appended — train.bin now {size_gb:.1f} GB")
    return tokens_written


# ═══════════════════════════════════════════════════════════
# PRETRAIN DATA  (~200GB total)
# ═══════════════════════════════════════════════════════════
def download_pretrain_data():
    pretrain_dir = os.path.join(BASE_DIR, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    print("=" * 70)
    print("  PRETRAINING DATA DOWNLOAD  (target: ~200GB)")
    print("=" * 70)

    # ── 1. OpenWebText (full: 8M docs, ~38GB) ──
    owt_path = os.path.join(pretrain_dir, "openwebtext")
    if not os.path.exists(owt_path):
        print("\n[1/6] Downloading OpenWebText (full, ~8M docs)...")
        ds = load_dataset("Skylion007/openwebtext", split="train", cache_dir=CACHE_DIR)
        ds.save_to_disk(owt_path)
        print(f"  ✓ Saved {len(ds):,} examples")
    else:
        print(f"\n[1/6] ✓ OpenWebText exists ({owt_path})")

    # ── 2. TheStack Python (5M files, ~80GB) ──
    print("\n[2/6] TheStack Python...")
    stream_and_save(
        "the-stack-python",
        "bigcode/the-stack",
        os.path.join(pretrain_dir, "code_python"),
        n_samples=5_000_000,
        desc="TheStack Python (5M code files)",
        hf_kwargs={"data_dir": "data/python", "split": "train"},
    )

    # ── 3. OpenWebMath (full: 6.3M pages, ~30GB) ──
    print("\n[3/6] OpenWebMath...")
    stream_and_save(
        "openwebmath",
        "open-web-math/open-web-math",
        os.path.join(pretrain_dir, "openwebmath"),
        n_samples=6_500_000,
        desc="OpenWebMath (full, ~6.3M math pages)",
        hf_kwargs={"split": "train"},
    )

    # ── 4. MetaMathQA (full: 395K) ──
    print("\n[4/6] MetaMathQA...")
    download_full(
        "meta-math/MetaMathQA",
        os.path.join(pretrain_dir, "metamathqa"),
        "metamathqa",
        desc="MetaMathQA (395K math reasoning)",
        hf_kwargs={"split": "train"},
    )

    # ── 5. FineWeb-Edu — 10BT high-quality educational / STEM text ──────────
    # allenai/peS2o and EleutherAI/proof-pile-2 both rely on legacy loading
    # scripts incompatible with datasets >= 3.0.  FineWeb-Edu (sample-10BT)
    # is parquet-native, STEM-rich, and excellent for math/science pretraining.
    print("\n[5/6] FineWeb-Edu (educational/STEM text)...")
    stream_and_save(
        "fineweb-edu",
        "HuggingFaceFW/fineweb-edu",
        os.path.join(pretrain_dir, "fineweb_edu"),
        n_samples=2_000_000,
        desc="FineWeb-Edu (2M educational docs, 10BT subset)",
        hf_kwargs={"name": "sample-10BT", "split": "train"},
    )

    # ── 6. Wikipedia EN (full: ~6.4M articles) ──
    print("\n[6/9] Wikipedia...")
    wiki_path = os.path.join(pretrain_dir, "wikipedia")
    if not os.path.exists(wiki_path):
        print("  ↓ Downloading Wikipedia EN (full, ~6.4M articles)...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                              split="train", cache_dir=CACHE_DIR, token=_hf_token)
            ds.save_to_disk(wiki_path)
            print(f"  ✓ Saved {len(ds):,} articles")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print(f"  ✓ Wikipedia exists ({wiki_path})")

    # ── 7. C4 English (3M docs, ~12GB) ── great general-web complement to OWT ──
    print("\n[7/9] C4 English (3M docs)...")
    stream_and_save(
        "c4_en",
        "allenai/c4",
        os.path.join(pretrain_dir, "c4_en"),
        n_samples=3_000_000,
        desc="C4 English (3M docs, filtered web text)",
        hf_kwargs={"name": "en", "split": "train"},
    )

    # ── 8. CC-News (1M news articles, long-form diverse text) ─────────────────
    print("\n[8/9] CC-News (1M articles)...")
    stream_and_save(
        "cc_news",
        "cc_news",
        os.path.join(pretrain_dir, "cc_news"),
        n_samples=1_000_000,
        desc="CC-News (1M news articles)",
        hf_kwargs={"split": "train"},
    )

    # ── 9. TheStack Java + JavaScript (2M each) ────────────────────────────────
    print("\n[9/9] TheStack Java + JavaScript...")
    for lang, field_dir in [("java", "data/java"), ("javascript", "data/javascript")]:
        stream_and_save(
            f"code_{lang}",
            "bigcode/the-stack",
            os.path.join(pretrain_dir, f"code_{lang}"),
            n_samples=2_000_000,
            desc=f"TheStack {lang.capitalize()} (2M files)",
            hf_kwargs={"data_dir": field_dir, "split": "train"},
        )

    # ── Summary ──
    print("\n" + "─" * 70)
    print("  PRETRAIN DATA SUMMARY:")
    for d in sorted(os.listdir(pretrain_dir)):
        full = os.path.join(pretrain_dir, d)
        if os.path.isdir(full) and d not in ("cache", "hf_cache"):
            try:
                size = sum(
                    os.path.getsize(os.path.join(full, f))
                    for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))
                ) / (1024**3)
                print(f"    {d:30s}  {size:>8.1f} GB")
            except Exception:
                print(f"    {d:30s}  (error reading size)")
    print("─" * 70)
    print("  ✓ Pretraining data download complete!\n")


# ═══════════════════════════════════════════════════════════
# NEW PRETRAIN DATA  (run after download_pretrain_data)
# CodeSearchNet · GitHub Code · PG-19 Books · peS2o Papers · StackExchange
# All sources get: quality filtering, English detection, MD5 dedup,
# provenance metadata (__source__).
# ═══════════════════════════════════════════════════════════

def download_new_pretrain_data():
    pretrain_dir = os.path.join(BASE_DIR, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    print("=" * 70)
    print("  NEW PRETRAINING DATA DOWNLOAD")
    print("  (CodeSearchNet · GitHub Code · ThePile-Books · ThePile-Math · UltraChat · FLAN · SmolTalk · Alpaca)")
    print("=" * 70)

    # ── 1. CodeSearchNet — structured docstring + function (2M, py/java/js) ──
    # Each item: func_documentation_string + func_code_string → content field
    print("\n[1/5] CodeSearchNet (structured code + docstrings, 2M)...")

    def _csn_transform(item):
        lang = (item.get("language") or "").lower()
        if lang not in {"python", "java", "javascript"}:
            return None
        code = (item.get("func_code_string") or "").strip()
        if not code:
            return None
        if lang == "python":
            if not code_quality_python(code, max_lines=300):
                return None
        else:
            if not code_quality_generic(code, max_lines=400):
                return None
        if is_duplicate(code):
            return None
        doc = (item.get("func_documentation_string") or "").strip()
        content = f"{doc}\n\n{code}" if doc else code
        return {"content": content, "language": lang, "__source__": "code_search_net"}

    reset_dedup()
    stream_and_save(
        "code_search_net",
        "code_search_net",
        os.path.join(pretrain_dir, "code_search_net"),
        n_samples=2_000_000,
        desc="CodeSearchNet (py/java/js, docstring+function)",
        hf_kwargs={"split": "train"},
        transform_fn=_csn_transform,
    )

    # ── 2. bigcode/the-stack-smol — 300k code across 30 languages ────────────
    # 10k samples per language: TypeScript, Rust, Go, C, C++, SQL, Shell, Ruby,
    # Scala, PHP, etc.  Pure parquet, no loading script.
    # We already have 5M Python + 2M Java/JS from TheStack, so we keep ALL
    # 30 languages here for breadth; Python/Java/JS add 30k more filtered rows.
    # Fields: content, lang, avg_line_length, alphanum_fraction.
    print("\n[2/5] The-Stack-Smol full (300K, 30 languages)...")

    def _gh_filter(item):
        code = item.get("content") or ""
        if not code:
            return False
        lang = (item.get("lang") or "").lower()
        if lang == "python":
            return code_quality_python(code)
        return code_quality_generic(code, min_lines=3, max_lines=600)

    def _gh_transform(item):
        code = item.get("content") or ""
        if is_duplicate(code):
            return None
        return {
            "content": code,
            "language": item.get("lang") or "",
            "__source__": "code_github",
        }

    reset_dedup()
    stream_and_save(
        "code_github",
        "bigcode/the-stack-smol",
        os.path.join(pretrain_dir, "code_github"),
        n_samples=300_000,
        desc="The-Stack-Smol (bigcode/the-stack-smol, 30 languages)",
        hf_kwargs={"split": "train"},
        filter_fn=_gh_filter,
        transform_fn=_gh_transform,
    )

    # ── 3. Long-form text — EleutherAI/the_pile_deduplicated (books/web mix) ──
    # The Pile deduplicated has books, web, academic, code and more with the
    # loading-script-free parquet interface.  We keep long docs (≥800 chars)
    # as a proxy for book/essay-length content.
    print("\n[3/5] The Pile deduplicated (long-form slice, 200K)...")

    def _book_filter(item):
        text = item.get("text", "")
        return (text_quality(text, min_len=800)
                and is_english(text)
                and not is_duplicate(text))

    def _book_transform(item):
        return {"text": item["text"], "__source__": "redpajama_books"}

    reset_dedup()
    stream_and_save(
        "redpajama_books",
        "EleutherAI/the_pile_deduplicated",
        os.path.join(pretrain_dir, "redpajama_books"),
        n_samples=200_000,
        desc="The Pile deduplicated (long-form text slice)",
        hf_kwargs={"split": "train"},
        filter_fn=_book_filter,
        transform_fn=_book_transform,
    )

    # ── 4. Math/science — EleutherAI/the_pile_deduplicated (math quality) ────
    # Same corpus, different quality gate: equation/symbol density selects for
    # mathematical/scientific writing (ArXiv, MathExchange, etc. are in mix).
    print("\n[4/5] The Pile deduplicated (math/science slice, 300K)...")

    def _arxiv_filter(item):
        text = item.get("text", "")
        return (len(text) >= 300
                and math_quality(text)
                and not is_duplicate(text))

    def _arxiv_transform(item):
        return {"text": item["text"], "__source__": "arxiv_math"}

    reset_dedup()
    stream_and_save(
        "arxiv_math",
        "EleutherAI/the_pile_deduplicated",
        os.path.join(pretrain_dir, "arxiv_math"),
        n_samples=300_000,
        desc="The Pile deduplicated (math/science slice)",
        hf_kwargs={"split": "train"},
        filter_fn=_arxiv_filter,
        transform_fn=_arxiv_transform,
    )

    # ── 5. Instruction Q&A — HuggingFaceH4/ultrachat_200k ────────────────────
    # 200K high-quality multi-turn chat conversations (UltraChat filtered).
    # Fields: prompt (str), messages (list of {role, content}).
    # We flatten the conversation into "Human: …\n\nAssistant: …" prose.
    print("\n[5/8] UltraChat 200K — HuggingFaceH4/ultrachat_200k (200K)...")

    def _se_transform(item):
        messages = item.get("messages") or []
        parts = []
        for m in messages:
            role    = (m.get("role") or "").strip().capitalize()
            content = (m.get("content") or "").strip()
            if role and content:
                parts.append(f"{role}: {content}")
        text = "\n\n".join(parts).strip()
        if not text_quality(text, min_len=150):
            return None
        if not is_english(text):
            return None
        if is_duplicate(text):
            return None
        return {"text": text, "__source__": "stackexchange"}

    reset_dedup()
    stream_and_save(
        "stackexchange",
        "HuggingFaceH4/ultrachat_200k",
        os.path.join(pretrain_dir, "stackexchange"),
        n_samples=200_000,          # pull the whole dataset
        desc="UltraChat 200K (HuggingFaceH4/ultrachat_200k)",
        hf_kwargs={"split": "train_sft"},
        transform_fn=_se_transform,
    )

    # ── 6. FLAN collection — Muennighoff/flan ───────────────────────────────
    # 15M FLAN examples (templates over NLP tasks).  We pull 500K and format
    # as "### Instruction:\n{inputs}\n\n### Response:\n{targets}" so the base
    # model sees instruction structure during pretraining.
    # Fields: inputs (str), targets (str), task (str)
    print("\n[6/8] FLAN collection — Muennighoff/flan (500K)...")

    def _flan_transform(item):
        inp  = (item.get("inputs")  or "").strip()
        tgt  = (item.get("targets") or "").strip()
        if not inp or not tgt:
            return None
        text = f"### Instruction:\n{inp}\n\n### Response:\n{tgt}"
        if not text_quality(text, min_len=80):
            return None
        if not is_english(text):
            return None
        if is_duplicate(text):
            return None
        return {"text": text, "__source__": "flan"}

    reset_dedup()
    stream_and_save(
        "flan",
        "Muennighoff/flan",
        os.path.join(pretrain_dir, "flan"),
        n_samples=500_000,
        desc="FLAN collection (500K diverse NLP task templates)",
        hf_kwargs={"split": "train"},
        transform_fn=_flan_transform,
    )

    # ── 7. SmolTalk — HuggingFaceTB/smoltalk ────────────────────────────────
    # ~20M multi-turn conversational pairs from SmolLM-2 data curation.
    # Fields: messages (list of {role, content}), source (str).
    # We flatten into "### Instruction:\n{user}\n\n### Response:\n{assistant}"
    # for each consecutive user/assistant turn pair.
    print("\n[7/8] SmolTalk — HuggingFaceTB/smoltalk (300K)...")

    def _smoltalk_transform(item):
        messages = item.get("messages") or []
        parts = []
        for i in range(len(messages) - 1):
            u = messages[i]
            a = messages[i + 1]
            if (u.get("role") or "").lower() == "user" and \
               (a.get("role") or "").lower() == "assistant":
                usr = (u.get("content") or "").strip()
                ast = (a.get("content") or "").strip()
                if usr and ast:
                    parts.append(f"### Instruction:\n{usr}\n\n### Response:\n{ast}")
        if not parts:
            return None
        text = "\n\n".join(parts)
        if not text_quality(text, min_len=100):
            return None
        if is_duplicate(text):
            return None
        return {"text": text, "__source__": "smoltalk"}

    reset_dedup()
    stream_and_save(
        "smoltalk",
        "HuggingFaceTB/smoltalk",
        os.path.join(pretrain_dir, "smoltalk"),
        n_samples=300_000,
        desc="SmolTalk (300K multi-turn conversational instruction pairs)",
        hf_kwargs={"split": "train"},
        transform_fn=_smoltalk_transform,
    )

    # ── 8. Alpaca — tatsu-lab/alpaca ─────────────────────────────────────────
    # 52K GPT-3.5 generated instruction-following pairs (full dataset).
    # Fields: instruction (str), input (str), output (str).
    # Formatted as "### Instruction:\n{instruction}[\n\n### Input:\n{input}]\n\n### Response:\n{output}".
    print("\n[8/8] Alpaca — tatsu-lab/alpaca (full, ~52K)...")

    def _alpaca_transform(item):
        instruction = (item.get("instruction") or "").strip()
        inp         = (item.get("input")       or "").strip()
        output      = (item.get("output")      or "").strip()
        if not instruction or not output:
            return None
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        if not text_quality(text, min_len=60):
            return None
        if is_duplicate(text):
            return None
        return {"text": text, "__source__": "alpaca"}

    reset_dedup()
    stream_and_save(
        "alpaca",
        "tatsu-lab/alpaca",
        os.path.join(pretrain_dir, "alpaca"),
        n_samples=52_002,          # full dataset
        desc="Alpaca (tatsu-lab/alpaca, 52K GPT-3.5 instruction pairs)",
        hf_kwargs={"split": "train"},
        transform_fn=_alpaca_transform,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    new_sources = ["code_search_net", "code_github", "redpajama_books",
                   "arxiv_math", "stackexchange", "flan", "smoltalk", "alpaca"]
    print("\n" + "─" * 70)
    print("  NEW PRETRAIN DATA SUMMARY:")
    for d in new_sources:
        full = os.path.join(pretrain_dir, d)
        if os.path.isdir(full):
            try:
                size = sum(
                    os.path.getsize(os.path.join(full, f))
                    for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))
                ) / (1024**3)
                print(f"    {d:30s}  {size:>8.1f} GB")
            except Exception:
                print(f"    {d:30s}  (error reading size)")
        else:
            print(f"    {d:30s}  NOT DOWNLOADED")
    print("─" * 70)
    print("  ✓ New pretraining data download complete!")
    print()
    print("  Next step — append-tokenize new sources:")
    print("    conda run -n deepfill python data/dataset.py \\")
    print("      --stage append \\")
    print("      --sources code_search_net,code_github,redpajama_books,arxiv_math,stackexchange,flan,smoltalk,alpaca")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONTAMINATION CHECK
# Hashes n-grams from benchmark test sets, scans pretrain Arrow files, reports
# overlap.  Does NOT automatically remove — user inspects and decides.
# ═══════════════════════════════════════════════════════════════════════════════

def check_benchmark_contamination(
    pretrain_dir: str = "/mnt/zone/A/datasets/pretrain",
    bench_dir: str = "/mnt/zone/A/datasets/benchmarks",
    ngram: int = 13,
    min_hits_to_flag: int = 3,
):
    """
    Check pretrain Arrow shards for near-duplicate overlap with benchmark test
    examples (HumanEval, GSM8K, MATH).

    Uses shingled n-gram fingerprints (default 13-grams of words).  Reports
    sources and approximate file counts that contain contamination; does not
    modify any data.
    """
    import json

    print("=" * 70)
    print(f"  BENCHMARK CONTAMINATION CHECK  (n-gram={ngram})")
    print("=" * 70)

    # ── Build fingerprint set from benchmark test examples ────────────────────
    bench_fingerprints: set[int] = set()

    def _extract_text(item):
        for f in ("prompt", "question", "problem", "task_id", "text"):
            v = item.get(f)
            if isinstance(v, str) and len(v) > 20:
                return v
        return ""

    def _shingle(text: str, n: int) -> list[int]:
        words = text.lower().split()
        return [
            hash(" ".join(words[i:i + n]))
            for i in range(max(0, len(words) - n + 1))
        ]

    bench_datasets = {}
    for name in ["humaneval", "gsm8k", "math"]:
        path = os.path.join(bench_dir, name)
        if os.path.exists(path):
            try:
                ds = load_from_disk(path)
                texts = [_extract_text(item) for item in ds]
                shingles = [h for t in texts for h in _shingle(t, ngram)]
                bench_fingerprints.update(shingles)
                bench_datasets[name] = len(texts)
                print(f"  Loaded {name}: {len(texts):,} examples → {len(shingles):,} shingles")
            except Exception as e:
                print(f"  ⚠ Could not load {name}: {e}")

    if not bench_fingerprints:
        print("  ✗ No benchmark datasets found. Run --stage benchmark to download them first.")
        return

    print(f"\n  Total benchmark fingerprints: {len(bench_fingerprints):,}")
    print(f"  Scanning pretrain sources for {ngram}-gram overlap...\n")

    # ── Scan each pretrain source ─────────────────────────────────────────────
    report = {}
    for source in sorted(os.listdir(pretrain_dir)):
        source_path = os.path.join(pretrain_dir, source)
        if not os.path.isdir(source_path):
            continue

        # Try to load as Arrow dataset
        try:
            ds = load_from_disk(source_path)
        except Exception:
            continue

        hit_count = 0
        flagged_ids = []
        text_fields = ["text", "content", "func_code_string"]

        for i, item in enumerate(ds):
            text = ""
            for f in text_fields:
                v = item.get(f, "")
                if isinstance(v, str) and len(v) > 50:
                    text = v
                    break
            if not text:
                continue
            shingles = _shingle(text, ngram)
            hits = sum(1 for s in shingles if s in bench_fingerprints)
            if hits >= min_hits_to_flag:
                hit_count += 1
                if len(flagged_ids) < 5:
                    flagged_ids.append(i)

        report[source] = hit_count
        status = f"  ⚠  {hit_count:,} docs flagged" if hit_count else "  ✓  clean"
        print(f"    {source:30s}  {status}")

    print("\n" + "─" * 70)
    total_flagged = sum(report.values())
    if total_flagged == 0:
        print("  ✓ No contamination detected across all pretrain sources.")
    else:
        print(f"  ⚠ Total flagged documents: {total_flagged:,}")
        print(f"    Review flagged sources manually before training.")
    print("─" * 70 + "\n")


# ═══════════════════════════════════════════════════════════
# SFT DATA
# ═══════════════════════════════════════════════════════════
def download_sft_data():
    sft_dir = os.path.join(BASE_DIR, "sft")
    os.makedirs(sft_dir, exist_ok=True)

    print("=" * 70)
    print("  SFT DATA DOWNLOAD  (target: ~5GB, ~1.5M high-quality examples)")
    print("=" * 70)

    # ── High-quality instruction datasets ──────────────────────────────────
    # OpenHermes 2.5:  1M GPT-4 generated instructions, best public SFT data
    # ShareGPT90K:     real multi-turn GPT-4 conversations
    # WizardLM Evol:   harder/longer instruction variants
    # Orca-Math:       200K GPT-4 math solutions with step-by-step reasoning
    # MetaMathQA:      already in pretrain but re-used in instruct format
    # Code Alpaca:     20K code instructions
    datasets_to_download = [
        (
            "openhermes",
            "teknium/OpenHermes-2.5",
            {"split": "train"},
            "OpenHermes 2.5 (1M GPT-4 instructions — best general SFT)",
        ),
        (
            "sharegpt",
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            {"data_files": "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"},
            "ShareGPT 90K (real multi-turn GPT-4 conversations)",
        ),
        (
            "orca_math",
            "microsoft/orca-math-word-problems-200k",
            {"split": "train"},
            "Orca-Math (200K GPT-4 math word problems with solutions)",
        ),
        (
            "wizardlm_evol",
            "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
            {"split": "train"},
            "WizardLM Evol-Instruct V2 (196K diverse, evolved instructions)",
        ),
        (
            "code_alpaca",
            "sahil2801/CodeAlpaca-20k",
            {"split": "train"},
            "Code Alpaca (20K code instructions)",
        ),
        (
            "dolly",
            "databricks/databricks-dolly-15k",
            {"split": "train"},
            "Dolly 15K (diverse human-written instructions)",
        ),
    ]

    for i, (name, hf_name, kwargs, desc) in enumerate(datasets_to_download, 1):
        out_path = os.path.join(sft_dir, name)
        if not os.path.exists(out_path):
            print(f"\n[{i}/{len(datasets_to_download)}] ↓ {desc}...")
            try:
                ds = load_dataset(hf_name, cache_dir=CACHE_DIR, token=_hf_token, **kwargs)
                ds.save_to_disk(out_path)
                n = len(ds) if not isinstance(ds, dict) else sum(len(v) for v in ds.values())
                print(f"  ✓ Saved {n:,} examples")
            except Exception as e:
                print(f"  ✗ Failed to download {hf_name}: {e}")
        else:
            print(f"\n[{i}/{len(datasets_to_download)}] ✓ {name} exists")

    print("\n  ✓ SFT data download complete!\n")


# ═══════════════════════════════════════════════════════════
# DPO DATA
# ═══════════════════════════════════════════════════════════
def download_dpo_data():
    dpo_dir = os.path.join(BASE_DIR, "dpo")
    os.makedirs(dpo_dir, exist_ok=True)

    print("=" * 70)
    print("  DPO DATA DOWNLOAD")
    print("=" * 70)

    dpo_datasets = [
        ("ultrafeedback", "HuggingFaceH4/ultrafeedback_binarized", {"split": "train_prefs"}, "UltraFeedback (61K prefs)"),
        ("orca_dpo",      "Intel/orca_dpo_pairs",                   {"split": "train"},       "Orca DPO Pairs (13K)"),
    ]

    for i, (name, hf_name, kwargs, desc) in enumerate(dpo_datasets, 1):
        out_path = os.path.join(dpo_dir, name)
        if not os.path.exists(out_path):
            print(f"\n[{i}/{len(dpo_datasets)}] ↓ {desc}...")
            try:
                ds = load_dataset(hf_name, cache_dir=CACHE_DIR, token=_hf_token, **kwargs)
                ds.save_to_disk(out_path)
                n = len(ds) if not isinstance(ds, dict) else sum(len(v) for v in ds.values())
                print(f"  ✓ Saved {n:,} examples")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"\n[{i}/{len(dpo_datasets)}] ✓ {name} exists")

    print("\n  ✓ DPO data download complete!\n")


# ═══════════════════════════════════════════════════════════
# BENCHMARK DATA
# ═══════════════════════════════════════════════════════════
def download_benchmark_data():
    bench_dir = os.path.join(BASE_DIR, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)

    print("=" * 70)
    print("  BENCHMARK DATA DOWNLOAD")
    print("=" * 70)

    benchmarks = [
        ("humaneval", "openai/openai_humaneval",          {"split": "test"}, "HumanEval"),
        ("mbpp",      "google-research-datasets/mbpp",    {"split": "test"}, "MBPP"),
        ("gsm8k",     "openai/gsm8k",        {"name": "main", "split": "test"}, "GSM8K"),
        ("math",      "lighteval/MATH",       {"name": "all",  "split": "test"}, "MATH"),
    ]

    for i, (name, hf_name, kwargs, desc) in enumerate(benchmarks, 1):
        out_path = os.path.join(bench_dir, name)
        if not os.path.exists(out_path):
            print(f"\n[{i}/{len(benchmarks)}] ↓ {desc}...")
            try:
                ds = load_dataset(hf_name, cache_dir=CACHE_DIR, token=_hf_token, **kwargs)
                ds.save_to_disk(out_path)
                n = len(ds) if not isinstance(ds, dict) else sum(len(v) for v in ds.values())
                print(f"  ✓ Saved {n:,} examples")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"\n[{i}/{len(benchmarks)}] ✓ {name} exists")

    print("\n  ✓ Benchmark data download complete!\n")


# ═══════════════════════════════════════════════════════════
# LARGE-SCALE PRETRAIN  (stream → uint32 → train.bin directly)
# ═══════════════════════════════════════════════════════════
def download_large_pretrain_data():
    """Stream-tokenise large corpora directly into train.bin (no Arrow files).

    Sources and full sizes (all disk-limited by 50 GB guard):
      FineWeb-350BT    : full 350 B tokens →  1.4 TB (disk-limited)
      Falcon RefinedWeb: full ~600 B tokens → ~2.4 TB (disk-limited)
      DCLM baseline    : full ~4 T tokens   → ~16 TB  (disk-limited)

    All three datasets are resumable: run the command again if interrupted and
    each source will fast-skip already-processed rows and continue appending.
    Stops automatically when disk has < 50 GB free.
    """
    pretrain_dir = os.path.join(BASE_DIR, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)
    train_bin = os.path.join(pretrain_dir, "train.bin")

    print("=" * 70)
    print("  LARGE PRETRAIN — stream → tokenize → train.bin  (no Arrow cache)")
    print("=" * 70)

    if not os.path.exists(train_bin):
        print(f"  ⚠  {train_bin} not found — will create a fresh bin file.")
    else:
        cur_gb  = os.path.getsize(train_bin) / (1024 ** 3)
        cur_tok = os.path.getsize(train_bin) // 4
        print(f"  Current train.bin: {cur_gb:.1f} GB  ({cur_tok/1e9:.1f}B tokens)")

    # ── Free-disk sanity check ────────────────────────────────────────────────
    import shutil as _shutil
    stat = _shutil.disk_usage(pretrain_dir)
    free_gb = stat.free / (1024 ** 3)
    print(f"  Disk free on /mnt/zone/A: {free_gb:.0f} GB")
    if free_gb < 200:
        print("  ✗ Less than 200 GB free — aborting to protect disk space.")
        return

    print()
    total_new = 0

    # ── 1. FineWeb-350BT (high-quality CommonCrawl, deduplicated, edu-filtered)
    reset_dedup()
    total_new += stream_tokenize_to_bin(
        source_name     = "fineweb_350bt",
        hf_name         = "HuggingFaceFW/fineweb",
        text_field      = "text",
        bin_path        = train_bin,
        n_tokens_target = 350_000_000_000,   # full sample-350BT corpus (~350B tokens)
        hf_kwargs       = {"name": "sample-350BT", "split": "train"},
        filter_fn       = lambda x: (
            text_quality(x.get("text", ""))
            and not is_duplicate(x.get("text", ""))
        ),
    )

    # ── 2. Falcon RefinedWeb — DISABLED (600B tokens, ~2.4 TB)
    # Uncomment to resume. Requires ~2.4 TB free disk.
    # reset_dedup()
    # total_new += stream_tokenize_to_bin(
    #     source_name     = "falcon_refinedweb",
    #     hf_name         = "tiiuae/falcon-refinedweb",
    #     text_field      = "content",
    #     bin_path        = train_bin,
    #     n_tokens_target = 600_000_000_000,
    #     hf_kwargs       = {"split": "train"},
    #     filter_fn       = lambda x: (
    #         text_quality(x.get("content", ""), min_len=300)
    #         and is_english(x.get("content", ""))
    #         and not is_duplicate(x.get("content", ""))
    #     ),
    # )

    # ── 3. DCLM baseline — DISABLED (4T tokens, ~16 TB)
    # Uncomment to resume. Requires massive disk.
    # reset_dedup()
    # total_new += stream_tokenize_to_bin(
    #     source_name     = "dclm_baseline",
    #     hf_name         = "mlfoundations/dclm-baseline-1.0",
    #     text_field      = "text",
    #     bin_path        = train_bin,
    #     n_tokens_target = 4_000_000_000_000,
    #     hf_kwargs       = {"split": "train"},
    #     filter_fn       = lambda x: (
    #         text_quality(x.get("text", ""), min_len=300)
    #         and is_english(x.get("text", ""))
    #         and not is_duplicate(x.get("text", ""))
    #     ),
    # )

    final_gb  = os.path.getsize(train_bin) / (1024 ** 3) if os.path.exists(train_bin) else 0
    final_tok = os.path.getsize(train_bin) // 4 if os.path.exists(train_bin) else 0

    print(f"\n{'='*70}")
    print(f"  Large pretrain complete:")
    print(f"    New tokens appended : {total_new/1e9:.1f}B")
    print(f"    train.bin final size: {final_gb:.1f} GB  ({final_tok/1e9:.1f}B tokens)")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for NovaMind-3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  pretrain          Original 9 pretrain sources (OWT, TheStack, OpenWebMath, …)
  new_pretrain      NEW sources: CodeSearchNet, GitHub Code, RedPajama arXiv, RedPajama SE, lvwerra SE
  large_pretrain    LARGE sources: FineWeb-350BT (100B), Falcon (50B), DCLM (50B)
                    Streams directly into train.bin — no Arrow intermediates.
                    Fully resumable; run again to continue from last checkpoint.
  sft               Instruction-tuning datasets
  dpo               Preference-tuning datasets
  benchmark         HumanEval / GSM8K / MATH test sets
  check_contamination  Scan pretrain Arrow files for benchmark overlap
  all               pretrain + new_pretrain + sft + dpo + benchmark

After downloading new_pretrain, tokenize with:
  python data/dataset.py --stage append \\
    --sources code_search_net,code_github,redpajama_books,arxiv_math,stackexchange

For large_pretrain (no separate tokenise step needed — tokens already in train.bin):
  python data/download.py --stage large_pretrain
""",
    )
    parser.add_argument(
        "--stage",
        choices=["pretrain", "new_pretrain", "large_pretrain", "sft", "dpo",
                 "benchmark", "check_contamination", "all"],
        default="all",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download sources even if they already exist on disk."
    )
    args = parser.parse_args()

    if args.data_dir:
        BASE_DIR = args.data_dir

    if args.force:
        globals()['_FORCE'] = True

    t0 = time.time()

    if args.stage in ("pretrain", "all"):
        download_pretrain_data()
    if args.stage in ("new_pretrain", "all"):
        download_new_pretrain_data()
    if args.stage == "large_pretrain":           # not included in "all" — explicit only
        download_large_pretrain_data()
    if args.stage in ("sft", "all"):
        download_sft_data()
    if args.stage in ("dpo", "all"):
        download_dpo_data()
    if args.stage in ("benchmark", "all"):
        download_benchmark_data()
    if args.stage == "check_contamination":
        check_benchmark_contamination()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  All done in {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"{'='*70}")
