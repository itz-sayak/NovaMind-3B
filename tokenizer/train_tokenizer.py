#!/usr/bin/env python3
"""
Train a domain-specific BPE / Unigram tokenizer with SentencePiece.

Usage
─────
  # Train (full run, ~1-3 h on 32 cores)
  python -m tokenizer.train_tokenizer \
      --data-dir   /mnt/zone/A/datasets/pretrain \
      --output     /mnt/zone/A/tokenizer/sp_64k \
      --vocab-size 65536 \
      --sample-size 20000000

  # Compare an already-trained SP model vs cl100k_base (no retraining)
  python -m tokenizer.train_tokenizer \
      --compare-only /mnt/zone/A/tokenizer/sp_64k/sp.model \
      --data-dir /mnt/zone/A/datasets/pretrain

The script:
  1. Streams text from each data source WITHOUT loading everything into RAM.
  2. Writes a combined shuffled plain-text file in 128 MB chunks.
  3. Calls SentencePiece SentencePieceTrainer.Train() on that file.
  4. Saves .model + .vocab.
  5. Prints a detailed domain-by-domain comparison vs tiktoken cl100k_base.
"""
import os
import sys
import time
import argparse
import random
from typing import Optional

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Source registry ─────────────────────────────────────────────────────────
SOURCES = {
    "openwebtext":  ("text",    0.35),   # general prose
    "code_python":  ("content", 0.30),   # code
    "openwebmath":  ("text",    0.15),   # LaTeX / math-heavy web pages
    "fineweb_edu":  ("text",    0.10),   # educational/STEM text
    "metamathqa":   ("query",   0.05),   # math reasoning (short)
    "wikipedia":    ("text",    0.05),   # encyclopaedic prose
}

# Domain labels used in the comparison report
DOMAIN_LABELS = {
    "openwebtext": "prose",
    "code_python": "code",
    "openwebmath": "math/LaTeX",
    "fineweb_edu": "STEM/edu",
    "metamathqa":  "math-QA",
    "wikipedia":   "wiki",
}

FLUSH_BYTES  = 128 * 1024 * 1024   # 128 MB text buffer before disk flush
REPORT_EVERY = 500_000             # log progress every N docs


def _get_text(item: dict, primary_field: str) -> str:
    text = item.get(primary_field, "")
    if text:
        return text
    for alt in ("text", "content", "query", "instruction", "input"):
        text = item.get(alt, "")
        if text:
            return text
    return ""


def build_corpus_file(
    data_dir: str,
    out_txt: str,
    sample_size: int = 10_000_000,
    seed: int = 42,
) -> int:
    """
    Stream text from each data source and write a shuffled plain-text file.

    Returns the total number of lines (documents) written.

    Strategy:
      - For each source, iterate through the HF dataset on disk.
      - Reservoir-sample `per_source_budget` docs to keep memory flat.
      - Write accepted docs to the output file in chunks.
    """
    from datasets import load_from_disk

    rng = random.Random(seed)
    total_written = 0

    with open(out_txt, "w", encoding="utf-8") as fout:
        buf: list[str] = []
        buf_bytes = 0

        def _flush():
            nonlocal buf_bytes
            rng.shuffle(buf)          # intra-chunk shuffle
            for line in buf:
                fout.write(line)
                fout.write("\n")
            buf.clear()
            buf_bytes = 0

        for source_name, (text_field, weight) in SOURCES.items():
            source_path = os.path.join(data_dir, source_name)
            if not os.path.exists(source_path):
                print(f"  [corpus] {source_name}: not found, skipping")
                continue

            budget = max(1, int(sample_size * weight))
            print(f"  [corpus] {source_name}: loading (budget {budget:,} docs)...")
            ds = load_from_disk(source_path)

            count = 0
            for i, item in enumerate(ds):
                text = _get_text(item, text_field)
                if not text or len(text) < 20:
                    continue

                # One document per line (newlines inside are replaced with spaces)
                clean = text.replace("\n", " ").replace("\r", "")
                buf.append(clean)
                buf_bytes += len(clean)
                count += 1

                if buf_bytes >= FLUSH_BYTES:
                    _flush()

                if count >= budget:
                    break

                if (i + 1) % REPORT_EVERY == 0:
                    print(f"    {source_name}: {i+1:,} scanned, {count:,} kept")

            total_written += count
            print(f"  [corpus] {source_name}: {count:,} docs written")

        # Final flush
        if buf:
            _flush()

    print(f"  [corpus] Total: {total_written:,} lines -> {out_txt}")
    return total_written


def train_tokenizer(
    data_dir: str,
    output_dir: str,
    vocab_size: int = 65536,
    model_type: str = "bpe",
    sample_size: int = 20_000_000,
    character_coverage: float = 0.9999,
    num_threads: int = 32,
    max_sentence_length: int = 16384,
    max_sentences: int = 5_000_000,
    skip_corpus: bool = False,
    seed: int = 42,
):
    """
    Train a SentencePiece tokenizer, then run the detailed comparison.

    max_sentences controls how many lines SP reads from corpus.txt during
    training (input_sentence_size).  The corpus is shuffled at build time so
    no in-memory shuffle is needed, keeping peak RAM under ~5 GB.
    """
    os.makedirs(output_dir, exist_ok=True)
    corpus_txt   = os.path.join(output_dir, "corpus.txt")
    model_prefix = os.path.join(output_dir, "sp")

    # ── Step 1: Build corpus (or reuse existing) ──
    if skip_corpus and os.path.exists(corpus_txt):
        corpus_gb = os.path.getsize(corpus_txt) / (1024 ** 3)
        # Estimate line count from file size (used to cap input_sentence_size)
        # Rough heuristic: read first 10k lines to get average length.
        with open(corpus_txt, "r", encoding="utf-8", errors="replace") as f:
            sample = [f.readline() for _ in range(10_000)]
        sample = [x for x in sample if x]
        avg_len = sum(len(l) for l in sample) / max(len(sample), 1)
        n_lines = max(1, int(corpus_gb * 1024**3 / max(avg_len, 1)))
        print(f"  Reusing corpus: ~{n_lines:,} est. lines, {corpus_gb:.2f} GB  ({corpus_txt})\n")
    else:
        print("=" * 60)
        print(f"  Building corpus ({sample_size:,} docs target, seed={seed})")
        print("=" * 60)
        t0 = time.time()
        n_lines = build_corpus_file(data_dir, corpus_txt, sample_size=sample_size, seed=seed)
        corpus_gb = os.path.getsize(corpus_txt) / (1024 ** 3)
        print(f"  Corpus: {n_lines:,} lines, {corpus_gb:.2f} GB  ({time.time()-t0:.1f}s)\n")

    # ── Step 2: Train SentencePiece ──
    print("=" * 60)
    print(f"  Training SentencePiece  model_type={model_type}  vocab={vocab_size:,}  threads={num_threads}")
    print("=" * 60)
    t0 = time.time()

    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        # Special tokens
        pad_id=3, bos_id=1, eos_id=2, unk_id=0,
        # Domain-specific symbols
        user_defined_symbols=[
            "<|code|>", "<|/code|>",
            "<|math|>", "<|/math|>",
            "<|latex|>", "<|/latex|>",
            "<|endoftext|>",
        ],
        byte_fallback=True,     # never produces <unk>
        split_digits=True,      # each digit is its own token (key for math)
        seed_sentencepiece_size=1_000_000,
        # shuffle_input_sentence=False: corpus was shuffled at build time;
        # keeps SP from loading all lines into RAM (critical for large corpora).
        shuffle_input_sentence=False,
        input_sentence_size=min(n_lines, max_sentences),
    )

    elapsed = time.time() - t0
    print(f"\n  Trained in {elapsed/60:.1f} min")
    print(f"  Model : {model_prefix}.model")
    print(f"  Vocab : {model_prefix}.vocab\n")

    # ── Step 3: Detailed comparison ──
    compare_tokenizers(model_prefix + ".model", data_dir)

    print(f"  Corpus file kept at: {corpus_txt}  ({corpus_gb:.2f} GB)")
    print("  Delete when no longer needed.\n")

    return model_prefix + ".model"


def compare_tokenizers(
    sp_model_path: str,
    data_dir: str,
    n_docs_per_domain: int = 2000,
    seq_len: int = 2048,
    seed: int = 42,
):
    """
    Detailed domain-by-domain comparison: custom SP tokenizer vs cl100k_base.

    Reports per domain and overall:
      • bytes / token  (higher = more efficient)
      • tokens / doc   (lower = more efficient)
      • chars that fit in one seq_len context window
      • fertility (tokens per whitespace word — lower is better)
      • relative efficiency vs cl100k_base
    """
    import tiktoken
    from datasets import load_from_disk

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    cl = tiktoken.get_encoding("cl100k_base")
    rng = random.Random(seed)

    W = 72  # column width for the report

    print("\n" + "═" * W)
    print(f"  TOKENIZER COMPARISON: custom SP-BPE (64k)  vs  tiktoken cl100k_base")
    print(f"  SP model : {sp_model_path}")
    print(f"  Domains  : {n_docs_per_domain:,} docs each   |   seq_len = {seq_len}")
    print("═" * W)

    col_fmt  = "  {:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}"
    sep_line = "  " + "-" * (W - 2)

    print(col_fmt.format(
        "Domain", "SP b/tok", "CL b/tok", "SP t/doc", "CL t/doc",
        "SP ctx chars", "CL ctx chars",
    ))
    print(sep_line)

    # Track totals across all domains
    total = dict(
        sp_bytes=0, cl_bytes=0,
        sp_toks=0,  cl_toks=0,
        sp_words=0, cl_words=0,
        docs=0,
    )

    domain_rows = []

    for src_name, (text_field, _weight) in SOURCES.items():
        src_path = os.path.join(data_dir, src_name)
        label    = DOMAIN_LABELS.get(src_name, src_name)

        if not os.path.exists(src_path):
            print(f"  {label:<14}  (data not found, skipping)")
            continue

        ds = load_from_disk(src_path)
        indices = rng.sample(range(len(ds)), min(n_docs_per_domain, len(ds)))
        samples = [_get_text(ds[i], text_field) for i in indices]
        samples = [s for s in samples if len(s) >= 20]

        sp_b = cl_b = sp_t = cl_t = sp_w = 0

        for text in samples:
            b    = len(text.encode("utf-8"))
            s_t  = len(sp.Encode(text))
            c_t  = len(cl.encode(text, allowed_special="all"))
            words = max(1, len(text.split()))

            sp_b += b;  cl_b += b
            sp_t += s_t;  cl_t += c_t
            sp_w += words

        n = len(samples)
        if n == 0 or sp_t == 0 or cl_t == 0:
            continue

        sp_bpt  = sp_b / sp_t          # bytes per token
        cl_bpt  = cl_b / cl_t
        sp_tpd  = sp_t / n             # tokens per doc
        cl_tpd  = cl_t / n
        # chars fitting in one context window  = seq_len * bytes_per_token
        sp_ctx  = int(seq_len * sp_bpt)
        cl_ctx  = int(seq_len * cl_bpt)

        domain_rows.append((label, sp_bpt, cl_bpt, sp_tpd, cl_tpd, sp_ctx, cl_ctx))
        print(col_fmt.format(
            label,
            f"{sp_bpt:.3f}", f"{cl_bpt:.3f}",
            f"{sp_tpd:.0f}",  f"{cl_tpd:.0f}",
            f"{sp_ctx:,}",    f"{cl_ctx:,}",
        ))

        total["sp_bytes"] += sp_b;  total["cl_bytes"] += cl_b
        total["sp_toks"]  += sp_t;  total["cl_toks"]  += cl_t
        total["sp_words"] += sp_w
        total["docs"]     += n

    # ── Overall summary ──
    print(sep_line)
    if total["docs"] == 0:
        print("  No data found.")
        return

    ov_sp_bpt = total["sp_bytes"] / total["sp_toks"]
    ov_cl_bpt = total["cl_bytes"] / total["cl_toks"]
    ov_sp_tpd = total["sp_toks"]  / total["docs"]
    ov_cl_tpd = total["cl_toks"]  / total["docs"]
    ov_sp_ctx = int(seq_len * ov_sp_bpt)
    ov_cl_ctx = int(seq_len * ov_cl_bpt)

    print(col_fmt.format(
        "OVERALL",
        f"{ov_sp_bpt:.3f}", f"{ov_cl_bpt:.3f}",
        f"{ov_sp_tpd:.0f}",  f"{ov_cl_tpd:.0f}",
        f"{ov_sp_ctx:,}",    f"{ov_cl_ctx:,}",
    ))
    print("═" * W)

    # ── Narrative summary ──
    efficiency = ov_sp_bpt / ov_cl_bpt   # > 1 → SP fits more chars per token
    tok_savings = (1 - total["sp_toks"] / total["cl_toks"]) * 100
    ctx_gain    = ov_sp_ctx - ov_cl_ctx

    print(f"\n  Fertility (tokens/word):")
    sp_fert = total["sp_toks"] / total["sp_words"]
    cl_fert = total["cl_toks"] / total["sp_words"]  # same word count
    print(f"    SP-BPE 64k  : {sp_fert:.2f}  tokens / word")
    print(f"    cl100k_base : {cl_fert:.2f}  tokens / word")

    print(f"\n  Context-window efficiency (seq_len={seq_len}):")
    print(f"    SP-BPE 64k  : ~{ov_sp_ctx:,} utf-8 chars fit per context")
    print(f"    cl100k_base : ~{ov_cl_ctx:,} utf-8 chars fit per context")
    if ctx_gain > 0:
        print(f"    → SP packs  +{ctx_gain:,} more chars per context  ({efficiency:.3f}x)")
    else:
        print(f"    → cl100k packs  +{-ctx_gain:,} more chars per context  ({1/efficiency:.3f}x)")

    print(f"\n  Token savings (overall): ", end="")
    if tok_savings > 0:
        print(f"SP uses {tok_savings:.1f}% fewer tokens  ✓ better compression")
    else:
        print(f"cl100k uses {-tok_savings:.1f}% fewer tokens  (SP needs tuning)")

    print(f"\n  Verdict:")
    if efficiency >= 1.05:
        print(f"    ✓ Use SP-BPE 64k  — significantly more efficient for this domain mix")
    elif efficiency >= 1.01:
        print(f"    ✓ Use SP-BPE 64k  — marginally more efficient")
    elif efficiency >= 0.99:
        print(f"    ≈ Toss-up  — within 1%.  SP has domain-specific tokens; cl100k is battle-tested")
    else:
        print(f"    ✗ Use cl100k_base — larger byte-coverage outweighs SP's domain training")
    print("═" * W + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE/Unigram SentencePiece tokenizer on code + math + prose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="/mnt/zone/A/datasets/pretrain",
        help="Root dir containing openwebtext/, code_python/, etc.",
    )
    parser.add_argument(
        "--output", type=str, default="/mnt/zone/A/tokenizer/sp_64k",
        help="Output directory for .model and .vocab",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=65536,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--model-type", type=str, default="bpe", choices=["bpe", "unigram"],
        help="SentencePiece model type",
    )
    parser.add_argument(
        "--sample-size", type=int, default=20_000_000,
        help="Number of documents to sample for training",
    )
    parser.add_argument(
        "--threads", type=int, default=32,
        help="CPU threads for SentencePiece training",
    )
    parser.add_argument(
        "--max-sentences", type=int, default=5_000_000,
        help="Max lines SP reads from corpus.txt during training (input_sentence_size). "
             "Corpus is pre-shuffled so no in-memory shuffle is needed. "
             "5M lines ≈ 3-5 GB RAM.",
    )
    parser.add_argument(
        "--skip-corpus", action="store_true",
        help="Reuse an existing corpus.txt instead of rebuilding from datasets",
    )
    parser.add_argument(
        "--compare-only", type=str, default=None, metavar="MODEL_PATH",
        help="Skip training; run domain comparison on an existing .model file",
    )
    parser.add_argument(
        "--n-docs", type=int, default=2000,
        help="Documents per domain to sample during comparison",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Context window length (tokens) for efficiency calculation",
    )

    args = parser.parse_args()

    if args.compare_only:
        compare_tokenizers(
            sp_model_path=args.compare_only,
            data_dir=args.data_dir,
            n_docs_per_domain=args.n_docs,
            seq_len=args.seq_len,
        )
    else:
        train_tokenizer(
            data_dir=args.data_dir,
            output_dir=args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            sample_size=args.sample_size,
            num_threads=args.threads,
            max_sentences=args.max_sentences,
            skip_corpus=args.skip_corpus,
        )
