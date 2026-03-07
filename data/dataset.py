"""
Dataset classes for pretraining, SFT, and DPO.

Pretraining: Packs documents into fixed-length sequences for efficient training.
SFT: Formats instruction-response pairs with proper masking.
DPO: Provides (prompt, chosen, rejected) triples for preference optimization.
"""
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# Add project root so `tokenizer` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.tokenizer import Tokenizer, get_tokenizer


# ── FIM (Fill-in-Middle) ─────────────────────────────────────────────────────
# Special token IDs in unused cl100k_base vocabulary slots (100277 → 100351)
FIM_BEGIN = 100278
FIM_HOLE  = 100279
FIM_END   = 100280

# Sources that should receive FIM augmentation (code token sequences)
_CODE_SOURCES = {
    "code_python", "code_java", "code_javascript",
    "code_search_net", "code_github",
}


def apply_fim_tokens(tokens, eos_id, fim_rate=0.1, rng=None):
    """Apply Fill-in-Middle (PSM format) to a token sequence.

    With probability ``fim_rate``, transforms tokens into
    Prefix-Suffix-Middle order using three special delimiters.

    Args:
        tokens:   list[int] – token IDs (with EOS at end).
        eos_id:   int – EOS token ID.
        fim_rate: float – probability of applying FIM per document.
        rng:      random.Random instance (for reproducibility).

    Returns:
        Possibly-transformed token list.
    """
    if rng is None:
        rng = random.Random()
    if rng.random() > fim_rate:
        return tokens

    # Strip trailing EOS
    if tokens and tokens[-1] == eos_id:
        core, has_eos = tokens[:-1], True
    else:
        core, has_eos = list(tokens), False

    if len(core) < 10:          # too short for meaningful FIM
        return tokens

    split = rng.randint(1, len(core) - 1)
    prefix = core[:split]
    rest   = core[split:]

    # PSM: randomly split rest into middle + suffix
    if len(rest) > 5 and rng.random() < 0.5:
        mid_split = rng.randint(1, len(rest) - 1)
        middle = rest[:mid_split]
        suffix = rest[mid_split:]
    else:
        middle = rest
        suffix = []

    result = [FIM_BEGIN] + prefix + [FIM_HOLE] + suffix + [FIM_END] + middle
    if has_eos:
        result.append(eos_id)
    return result


class PretrainDataset(Dataset):
    """
    Memory-mapped pretraining dataset.
    
    Loads pre-tokenized data from binary files and provides fixed-length windows.
    """
    
    def __init__(self, data_dir, seq_len=2048, split="train"):
        self.seq_len = seq_len
        
        # Look for pre-tokenized binary file
        bin_file = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(bin_file):
            self.data = np.memmap(bin_file, dtype=np.uint32, mode='r')
            print(f"Loaded {split} data: {len(self.data):,} tokens from {bin_file}")
        else:
            raise FileNotFoundError(
                f"Pre-tokenized data not found at {bin_file}. "
                f"Run tokenize_pretrain_data() first."
            )
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target offset
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]   # input
        y = chunk[1:]     # target
        return x, y


class StreamingPretrainDataset(IterableDataset):
    """Memory-efficient streaming pretraining dataset.

    Reads tokens sequentially from a memory-mapped binary file and yields
    fixed-length (seq_len, seq_len) input/target pairs.  To provide local
    shuffling without allocating a full-dataset random permutation (which
    would require ~50 GB for 354B tokens), it maintains a ring buffer of
    ``shuffle_buffer`` sequences and samples from it uniformly.

    Suitable for multi-worker DataLoaders and DDP: each (rank, worker) pair
    is assigned a non-overlapping shard of the data so that all tokens are
    covered without duplication across processes.

    Args:
        data_dir:        directory containing ``{split}.bin``.
        seq_len:         number of input tokens per sample.
        split:           ``"train"`` or ``"val"``.
        shuffle_buffer:  number of sequences to hold in the shuffle ring
                         buffer.  Set to 1 to disable shuffling.
        world_size:      total number of DDP ranks (sharding denominator).
        rank:            this process's DDP rank (sharding offset).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 2048,
        split: str = "train",
        shuffle_buffer: int = 2048,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.world_size = world_size
        self.rank = rank

        bin_file = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(bin_file):
            raise FileNotFoundError(
                f"Pre-tokenized data not found at {bin_file}. "
                f"Run tokenize_pretrain_data() first."
            )
        self._bin_file = bin_file
        # Open mmap just to report size; workers re-open independently to
        # avoid sharing mmap file descriptors across fork boundaries.
        _tmp = np.memmap(bin_file, dtype=np.uint32, mode='r')
        self._num_tokens = len(_tmp)
        del _tmp
        self._num_seqs = (self._num_tokens - 1) // self.seq_len
        print(f"Loaded {split} data: {self._num_tokens:,} tokens from {bin_file}")

    # DataLoader calls __len__ to estimate epoch length for progress bars.
    def __len__(self) -> int:
        return self._num_seqs // self.world_size

    def __iter__(self):
        # Re-open mmap in each worker to avoid fd sharing across forks.
        data = np.memmap(self._bin_file, dtype=np.uint32, mode='r')

        # Determine worker shard within this rank's slice.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_global_workers = self.world_size * worker_info.num_workers
            global_worker_id  = self.rank * worker_info.num_workers + worker_info.id
        else:
            num_global_workers = self.world_size
            global_worker_id  = self.rank

        # Partition sequences evenly across workers; drop remainder.
        seqs_per_worker = self._num_seqs // num_global_workers
        start_seq = global_worker_id * seqs_per_worker
        end_seq   = start_seq + seqs_per_worker

        # Shuffle buffer: a list of pre-fetched (x, y) tensors.
        rng = random.Random(42 + global_worker_id)
        buf: list = []

        for seq_idx in range(start_seq, end_seq):
            s = seq_idx * self.seq_len
            chunk = torch.from_numpy(data[s : s + self.seq_len + 1].astype(np.int64))
            buf.append((chunk[:-1], chunk[1:]))

            if len(buf) >= self.shuffle_buffer:
                # Pop a random item from the buffer.
                pick = rng.randrange(len(buf))
                buf[-1], buf[pick] = buf[pick], buf[-1]
                yield buf.pop()

        # Drain remaining items in the buffer.
        rng.shuffle(buf)
        yield from buf


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset.

    Sources (loaded if present):
      openhermes     - 1M GPT-4 generated instructions (chatml format)
      sharegpt       - 90K real multi-turn GPT-4 conversations
      wizardlm_evol  - 196K evolved instructions
      orca_math      - 200K math word problems with solutions
      code_alpaca    - 20K code instructions
      dolly          - 15K human-written instructions

    Only response tokens contribute to the loss (prompt tokens masked).
    """

    # ChatML system prompt used by OpenHermes / WizardLM
    SYSTEM = "You are a helpful, harmless, and honest assistant."

    def __init__(self, data_dir, max_len=4096, split="train"):
        self.tokenizer = get_tokenizer()
        self.max_len = max_len
        self.examples: list[dict] = []

        sources = [
            "openhermes",
            "sharegpt",
            "wizardlm_evol",
            "orca_math",
            "code_alpaca",
            "dolly",
        ]
        for source_dir in sources:
            source_path = os.path.join(data_dir, source_dir)
            if os.path.exists(source_path):
                before = len(self.examples)
                self._load_source(source_path, source_dir)
                print(f"  {source_dir}: +{len(self.examples)-before:,} examples")

        # Shuffle and split
        random.seed(42)
        random.shuffle(self.examples)
        split_idx = int(0.97 * len(self.examples))
        if split == "train":
            self.examples = self.examples[:split_idx]
        else:
            self.examples = self.examples[split_idx:]

        print(f"SFT dataset ({split}): {len(self.examples):,} examples")

    # ── source loaders ──────────────────────────────────────────────────────

    def _load_source(self, path, source_type):
        from datasets import load_from_disk
        try:
            ds = load_from_disk(path)
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
            return

        loader = {
            "openhermes":    self._parse_openhermes,
            "sharegpt":      self._parse_sharegpt,
            "wizardlm_evol": self._parse_wizardlm,
            "orca_math":     self._parse_orca_math,
            "code_alpaca":   self._parse_alpaca_style,
            "dolly":         self._parse_dolly,
        }.get(source_type)

        if loader is None:
            return
        for item in ds:
            result = loader(item)
            if result:
                self.examples.append(result)

    def _make_example(self, prompt: str, response: str) -> dict | None:
        """Return dict or None if too short/empty."""
        if not prompt.strip() or not response.strip():
            return None
        return {"prompt": prompt, "response": response}

    def _parse_openhermes(self, item):
        """OpenHermes 2.5: ChatML format with system/user/assistant turns."""
        convs = item.get("conversations") or []
        if not convs:
            return None
        system = next((c["value"] for c in convs if c.get("from") == "system"), self.SYSTEM)
        turns = [c for c in convs if c.get("from") in ("human", "gpt")]
        if not turns:
            return None
        # Use first human turn as prompt, first gpt turn as response
        prompt_parts = [f"<|system|>\n{system}\n"]
        for c in turns:
            role = "<|user|>" if c["from"] == "human" else "<|assistant|>"
            prompt_parts.append(f"{role}\n{c['value']}\n")
        # Split at last assistant turn
        if turns[-1].get("from") != "gpt":
            return None
        response = turns[-1]["value"]
        prompt = "".join(prompt_parts[:-1]) + "<|assistant|>\n"
        return self._make_example(prompt, response)

    def _parse_sharegpt(self, item):
        """ShareGPT: list of {from: human/gpt, value: ...} under 'conversations'."""
        convs = item.get("conversations") or []
        # Filter to human/gpt only
        convs = [c for c in convs if c.get("from") in ("human", "gpt")]
        if len(convs) < 2:
            return None
        # Build multi-turn context ending at last gpt response
        if convs[-1].get("from") != "gpt":
            convs = convs[:-1]
        if not convs:
            return None
        response = convs[-1]["value"]
        prompt_turns = convs[:-1]
        parts = [f"<|system|>\n{self.SYSTEM}\n"]
        for c in prompt_turns:
            role = "<|user|>" if c["from"] == "human" else "<|assistant|>"
            parts.append(f"{role}\n{c['value']}\n")
        parts.append("<|assistant|>\n")
        return self._make_example("".join(parts), response)

    def _parse_wizardlm(self, item):
        """WizardLM Evol: {conversations: [{from, value}]} same as ShareGPT."""
        return self._parse_sharegpt(item)

    def _parse_orca_math(self, item):
        """Orca-Math: {question, answer}."""
        q = item.get("question", "")
        a = item.get("answer", "")
        if not q or not a:
            return None
        prompt = f"<|system|>\n{self.SYSTEM}\n<|user|>\n{q}\n<|assistant|>\n"
        return self._make_example(prompt, a)

    def _parse_alpaca_style(self, item):
        """CodeAlpaca / Stanford Alpaca: {instruction, input, output}."""
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        if not instruction or not output:
            return None
        body = f"{instruction}\n\n{inp}" if inp.strip() else instruction
        prompt = f"<|system|>\n{self.SYSTEM}\n<|user|>\n{body}\n<|assistant|>\n"
        return self._make_example(prompt, output)

    def _parse_dolly(self, item):
        """Dolly 15K: {instruction, context, response}."""
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")
        if not instruction or not response:
            return None
        body = f"{instruction}\n\nContext: {context}" if context.strip() else instruction
        prompt = f"<|system|>\n{self.SYSTEM}\n<|user|>\n{body}\n<|assistant|>\n"
        return self._make_example(prompt, response)

    # ── tokenization ────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt   = example["prompt"]
        response = example["response"]

        prompt_tokens   = self.tokenizer.encode(prompt,   add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        eos = self.tokenizer.eos_token_id
        all_tokens = prompt_tokens + response_tokens + [eos]
        all_tokens = all_tokens[:self.max_len]

        # Pad to max_len
        pad_len    = self.max_len - len(all_tokens)
        all_tokens = all_tokens + [eos] * pad_len

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        labels = tokens.clone()

        # Mask everything up to and including the last prompt token
        prompt_len = min(len(prompt_tokens), self.max_len)
        labels[:prompt_len] = -1

        # Also mask padding
        if pad_len > 0:
            labels[-pad_len:] = -1

        loss_mask = (labels != -1).long()
        labels[labels == -1] = 0  # safe for embedding lookup

        return {
            "input_ids":  tokens,
            "labels":     labels,
            "loss_mask":  loss_mask,
        }


class DPODataset(Dataset):
    """
    Direct Preference Optimization dataset.
    
    Provides (prompt, chosen_response, rejected_response) triples.
    """
    
    def __init__(self, data_dir, max_len=2048, split="train"):
        self.tokenizer = get_tokenizer()
        self.max_len = max_len
        self.examples = []
        
        uf_path = os.path.join(data_dir, "ultrafeedback")
        if os.path.exists(uf_path):
            self._load_ultrafeedback(uf_path)
        
        # Split
        random.seed(42)
        random.shuffle(self.examples)
        split_idx = int(0.95 * len(self.examples))
        if split == "train":
            self.examples = self.examples[:split_idx]
        else:
            self.examples = self.examples[split_idx:]
        
        print(f"DPO dataset ({split}): {len(self.examples)} examples")
    
    def _load_ultrafeedback(self, path):
        from datasets import load_from_disk
        
        try:
            ds = load_from_disk(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return
        
        for item in ds:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", [])
            rejected = item.get("rejected", [])
            
            # Extract text from chosen/rejected (list of dicts with "content")
            if isinstance(chosen, list):
                chosen_text = " ".join(
                    [m.get("content", "") for m in chosen if m.get("role") == "assistant"]
                )
            else:
                chosen_text = str(chosen)
            
            if isinstance(rejected, list):
                rejected_text = " ".join(
                    [m.get("content", "") for m in rejected if m.get("role") == "assistant"]
                )
            else:
                rejected_text = str(rejected)
            
            if prompt and chosen_text and rejected_text:
                self.examples.append({
                    "prompt": f"### Human:\n{prompt}\n\n### Assistant:\n",
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        def tokenize_pair(prompt, response):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            eos = self.tokenizer.eos_token_id
            tokens = [eos] + prompt_tokens + response_tokens + [eos]
            tokens = tokens[:self.max_len]
            prompt_len = min(1 + len(prompt_tokens), len(tokens))
            pad_len = self.max_len - len(tokens)
            tokens = tokens + [eos] * pad_len
            # Labels (same as tokens but shifted during loss computation)
            labels = list(tokens)
            # Mask: 1 for response+eos tokens, 0 for prompt/padding
            mask = [0] * prompt_len
            resp_len = len(tokens) - pad_len - prompt_len
            mask += [1] * resp_len
            mask += [0] * pad_len
            mask = mask[:self.max_len]
            return (torch.tensor(tokens, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.long))
        
        chosen_ids, chosen_labels, chosen_mask = tokenize_pair(example["prompt"], example["chosen"])
        rejected_ids, rejected_labels, rejected_mask = tokenize_pair(example["prompt"], example["rejected"])
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_mask": rejected_mask,
        }


def tokenize_pretrain_data(data_dir="/mnt/zone/A/datasets/pretrain",
                            output_dir="/mnt/zone/A/datasets/pretrain",
                            split="train", max_tokens=None,
                            flush_every: int = 500_000,
                            fim_rate: float = 0.0):
    """
    Tokenize raw text datasets into binary files for efficient pretraining.

    **Chunked approach** — never holds more than `flush_every` tokens in RAM.
    Tokens are appended to a temporary file in fixed-size uint32 chunks,
    then the final file is split into train/val.  Peak RSS stays ~200 MB.

    Combining 15 pretrain sources with proportional mixing:
      General web:  openwebtext  20%, c4_en           9%
      Code:         code_python  16%, code_java        5%, code_javascript 4%,
                    code_search_net 3%, code_github    2%
      Math:         openwebmath   9%, metamathqa       4%
      Education:    fineweb_edu   7%, wikipedia        8%
      Books/news:   cc_news       7%, redpajama_books  6%
      Science:      arxiv_math    5%, stackexchange    1%
    """
    from datasets import load_from_disk

    tokenizer = get_tokenizer()  # uses whatever is configured
    _fim_rng = random.Random(42)  # deterministic FIM for reproducibility

    sources = {
        # (text_field, mix_weight)
        # General web
        "openwebtext":     ("text",    0.20),
        "c4_en":           ("text",    0.09),
        # Code
        "code_python":     ("content", 0.16),
        "code_java":       ("content", 0.05),
        "code_javascript": ("content", 0.04),
        "code_search_net": ("content",          0.03),   # docstring + code pairs
        "code_github":     ("content", 0.02),            # filtered GitHub code
        # Math
        "openwebmath":     ("text",    0.09),
        "metamathqa":      ("query",   0.04),
        # Educational / encyclopaedic
        "fineweb_edu":     ("text",    0.07),
        "wikipedia":       ("text",    0.08),
        # Books / long-form
        "cc_news":         ("text",    0.07),
        "redpajama_books": ("text",    0.06),             # PG-19 / public domain books
        # Science / technical
        "arxiv_math":      ("text",    0.05),             # arXiv math papers
        "stackexchange":   ("text",    0.01),             # SE Q&A pairs
    }

    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, "_tokens_tmp.bin")
    total_tokens_written = 0

    # Open a raw binary file for append-mode streaming writes
    with open(tmp_path, "wb") as tmp_f:
        token_buf: list[int] = []

        def _flush():
            nonlocal total_tokens_written
            if not token_buf:
                return
            arr = np.array(token_buf, dtype=np.uint32)
            arr.tofile(tmp_f)
            total_tokens_written += len(token_buf)
            token_buf.clear()

        for source_name, (text_field, weight) in sources.items():
            source_path = os.path.join(data_dir, source_name)
            if not os.path.exists(source_path):
                print(f"Warning: {source_path} not found, skipping")
                continue

            print(f"Tokenizing {source_name}...")
            ds = load_from_disk(source_path)

            target_tokens = int(max_tokens * weight) if max_tokens else float("inf")
            source_count = 0

            for i, item in enumerate(ds):
                text = item.get(text_field, "")
                if not text:
                    for alt_field in ["text", "content", "query", "instruction"]:
                        text = item.get(alt_field, "")
                        if text:
                            break
                if not text:
                    continue

                tokens = tokenizer.encode_ordinary(text)
                tokens.append(tokenizer.eos_token_id)
                if fim_rate > 0 and source_name in _CODE_SOURCES:
                    tokens = apply_fim_tokens(tokens, tokenizer.eos_token_id,
                                              fim_rate, _fim_rng)
                token_buf.extend(tokens)
                source_count += len(tokens)

                # Flush chunk to disk when buffer is large enough
                if len(token_buf) >= flush_every:
                    _flush()

                if source_count >= target_tokens:
                    break

                if (i + 1) % 100_000 == 0:
                    print(f"  {source_name}: {i+1} docs, {source_count:,} tokens")

            # Flush remainder for this source
            _flush()
            print(f"  {source_name}: {source_count:,} tokens total")

    print(f"\nTotal tokens written: {total_tokens_written:,}")

    # ── Memory-mapped split into train / val ──
    all_data = np.memmap(tmp_path, dtype=np.uint32, mode="r")
    n = len(all_data)
    val_size = max(1000, n // 200)       # 0.5 % for val

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    # Write train split in chunks so we don't double-allocate
    train_len = n - val_size
    _COPY_CHUNK = 50_000_000  # 50 M tokens at a time (~100 MB)
    with open(train_path, "wb") as f:
        for start in range(0, train_len, _COPY_CHUNK):
            end = min(start + _COPY_CHUNK, train_len)
            np.array(all_data[start:end]).tofile(f)

    # Val is small — just copy
    np.array(all_data[-val_size:]).tofile(val_path)

    del all_data
    os.remove(tmp_path)

    print(f"Saved train: {train_len:,} tokens -> {train_path}")
    print(f"Saved val: {val_size:,} tokens -> {val_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase-specific tokenization for multi-stage data rebalancing
# ─────────────────────────────────────────────────────────────────────────────

# Default source weight profiles per training phase.
# Weights are normalised to 1.0 inside tokenize_phase() — use relative values.
PHASE_DEFAULTS = {
    "phase1": {
        # 0→75 %: heavy general web (45-55%) + code (20-25%) + books/math/edu
        "openwebtext":     0.22,
        "c4_en":           0.12,
        "cc_news":         0.08,
        # code
        "code_python":     0.10,
        "code_java":       0.03,
        "code_javascript": 0.03,
        "code_search_net": 0.04,  # structured docstring+fn
        "code_github":     0.02,  # raw filtered code
        # books
        "redpajama_books": 0.08,
        # math
        "openwebmath":     0.06,
        "metamathqa":      0.02,
        "arxiv_math":      0.03,
        # edu / QA
        "fineweb_edu":     0.06,
        "wikipedia":       0.06,
        "stackexchange":   0.05,
    },
    "phase2": {
        # 75→90 %: specialised push — code 30-35%, math 15-20%
        "openwebtext":     0.18,
        "c4_en":           0.08,
        "cc_news":         0.04,
        # code (raise)
        "code_python":     0.14,
        "code_java":       0.05,
        "code_javascript": 0.04,
        "code_search_net": 0.08,  # more structured code
        "code_github":     0.04,
        # books
        "redpajama_books": 0.03,
        # math (raise)
        "openwebmath":     0.10,
        "metamathqa":      0.05,
        "arxiv_math":      0.08,
        # edu / QA
        "fineweb_edu":     0.05,
        "wikipedia":       0.03,
        "stackexchange":   0.04,
    },
    "phase3": {
        # 90→100 %: highest-quality STEM + curated code only
        "code_python":     0.18,
        "code_java":       0.04,
        "code_javascript": 0.04,
        "code_search_net": 0.18,  # docstring-gated = highest signal
        "code_github":     0.06,
        # math
        "openwebmath":     0.16,
        "metamathqa":      0.08,
        "arxiv_math":      0.12,
        # edu / QA
        "fineweb_edu":     0.08,
        "wikipedia":       0.04,
        "stackexchange":   0.02,
    },
}


def tokenize_phase(phase_name: str,
                   data_dir: str = "/mnt/zone/A/datasets/pretrain",
                   output_base: str = "/mnt/zone/A/datasets/pretrain",
                   max_tokens: int | None = None,
                   fim_rate: float = 0.0,
                   source_weights: dict | None = None,
                   flush_every: int = 500_000):
    """Tokenize a training phase with custom source weights.

    Creates ``output_base/<phase_name>/train.bin`` and ``val.bin``.

    Args:
        phase_name:     e.g. "phase1", "phase2", "phase3"
        data_dir:       root dir containing raw HF Arrow datasets per source
        output_base:    parent dir; phase files go into a subdirectory
        max_tokens:     total tokens for this phase (distributed by weights)
        fim_rate:       FIM augmentation probability for code sources
        source_weights: {source_name: weight} — overrides PHASE_DEFAULTS
        flush_every:    flush buffer size
    """
    from datasets import load_from_disk

    weights = source_weights or PHASE_DEFAULTS.get(phase_name)
    if weights is None:
        raise ValueError(f"Unknown phase '{phase_name}' and no source_weights given. "
                         f"Available defaults: {list(PHASE_DEFAULTS)}")

    # Normalise weights to sum to 1
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    output_dir = os.path.join(output_base, phase_name)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer()
    _fim_rng = random.Random(42)
    tmp_path = os.path.join(output_dir, "_tokens_tmp.bin")
    total_tokens_written = 0

    print(f"=== Tokenizing {phase_name} ===")
    print(f"  FIM rate: {fim_rate}")
    print(f"  Sources: {list(weights.keys())}")
    if max_tokens:
        print(f"  Max tokens: {max_tokens:,}")

    with open(tmp_path, "wb") as tmp_f:
        token_buf: list[int] = []

        def _flush():
            nonlocal total_tokens_written
            if not token_buf:
                return
            arr = np.array(token_buf, dtype=np.uint32)
            arr.tofile(tmp_f)
            total_tokens_written += len(token_buf)
            token_buf.clear()

        for source_name, weight in weights.items():
            source_path = os.path.join(data_dir, source_name)
            if not os.path.exists(source_path):
                print(f"  SKIP {source_name}: not found")
                continue

            text_field = _SOURCE_FIELD.get(source_name, "text")
            ds = load_from_disk(source_path)
            target_tokens = int(max_tokens * weight) if max_tokens else float("inf")
            source_count = 0

            print(f"  {source_name} (weight={weight:.2f}, target={target_tokens:,})...")

            for i, item in enumerate(ds):
                text = item.get(text_field, "")
                if not text:
                    for alt in ["text", "content", "query", "instruction"]:
                        text = item.get(alt, "")
                        if text:
                            break
                if not text:
                    continue

                tokens = tokenizer.encode_ordinary(text)
                tokens.append(tokenizer.eos_token_id)
                if fim_rate > 0 and source_name in _CODE_SOURCES:
                    tokens = apply_fim_tokens(tokens, tokenizer.eos_token_id,
                                              fim_rate, _fim_rng)
                token_buf.extend(tokens)
                source_count += len(tokens)

                if len(token_buf) >= flush_every:
                    _flush()
                if source_count >= target_tokens:
                    break
                if (i + 1) % 100_000 == 0:
                    print(f"    {i+1} docs, {source_count:,} tokens")

            _flush()
            print(f"    {source_name}: {source_count:,} tokens total")

    print(f"\n  Total tokens: {total_tokens_written:,}")

    # Split into train / val
    all_data = np.memmap(tmp_path, dtype=np.uint32, mode="r")
    n = len(all_data)
    val_size = max(1000, n // 200)
    train_len = n - val_size

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    _CHUNK = 50_000_000
    with open(train_path, "wb") as f:
        for start in range(0, train_len, _CHUNK):
            end = min(start + _CHUNK, train_len)
            np.array(all_data[start:end]).tofile(f)
    np.array(all_data[-val_size:]).tofile(val_path)

    del all_data
    os.remove(tmp_path)

    print(f"  {phase_name}/train.bin: {train_len:,} tokens")
    print(f"  {phase_name}/val.bin:   {val_size:,} tokens")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Incremental tokenization: append new sources to existing train.bin / val.bin
# without re-processing what's already done.
# ─────────────────────────────────────────────────────────────────────────────

# Maps every source name we might ever append to its HF text field.
_SOURCE_FIELD = {
    # Original sources
    "c4_en":           "text",
    "cc_news":         "text",
    "code_java":       "content",
    "code_javascript": "content",
    "openwebtext":     "text",
    "openwebmath":     "text",
    "wikipedia":       "text",
    "fineweb_edu":     "text",
    "metamathqa":      "query",
    "code_python":     "content",
    "redpajama_books": "text",
    # New sources
    "code_search_net": "content",   # combined docstring+code saved to `content`
    "code_github":     "content",   # raw code saved to `content`
    "arxiv_math":      "text",
    "stackexchange":   "text",
    # Instruction sources (pretrain-formatted)
    "flan":            "text",
    "smoltalk":        "text",
    "alpaca":          "text",
}


def append_new_sources(
    sources_csv: str,
    data_dir: str = "/mnt/zone/A/datasets/pretrain",
    output_dir: str = "/mnt/zone/A/datasets/pretrain",
    flush_every: int = 500_000,
    val_fraction: float = 0.005,
):
    """
    Tokenize a list of new sources and APPEND their tokens directly to
    train.bin and val.bin — no tmp file, no double disk usage.

    Uses a val_counter to route every ~1/val_fraction-th flush to val.bin
    and the rest to train.bin.  This avoids accumulating a full tmp file
    before splitting (which would require 2× the token storage on disk).

    Args:
        sources_csv: comma-separated source names, e.g. 'c4_en,redpajama_books'
        data_dir:    directory containing the raw HF Arrow datasets
        output_dir:  directory containing (and where we update) train.bin / val.bin
        flush_every: token buffer size before flushing to disk
        val_fraction: fraction of new tokens to route to val split (default 0.5%)
    """
    from datasets import load_from_disk
    import shutil as _shutil

    source_names = [s.strip() for s in sources_csv.split(",") if s.strip()]
    if not source_names:
        print("No sources to append.")
        return

    tokenizer = get_tokenizer()
    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found. Run full tokenize first."
        )

    train_tokens_before = os.path.getsize(train_path) // 4
    val_tokens_before   = os.path.getsize(val_path) // 4 if os.path.exists(val_path) else 0
    print(f"Existing train: {train_tokens_before:,} tokens  ({train_tokens_before*4/1e9:.1f} GB)")
    print(f"Existing val:   {val_tokens_before:,} tokens")
    print(f"Appending sources: {source_names}\n")

    total_new_train = 0
    total_new_val   = 0

    # Val routing: send every N-th flush to val, rest to train.
    val_every   = max(1, round(1.0 / val_fraction))   # e.g. 200 for 0.5%
    flush_count = 0

    token_buf: list[int] = []

    def _flush():
        nonlocal total_new_train, total_new_val, flush_count
        if not token_buf:
            return
        arr = np.array(token_buf, dtype=np.uint32)
        flush_count += 1
        if flush_count % val_every == 0:
            with open(val_path, "ab") as f:
                arr.tofile(f)
            total_new_val += len(token_buf)
        else:
            with open(train_path, "ab") as f:
                arr.tofile(f)
            total_new_train += len(token_buf)
        token_buf.clear()

        # Disk-space guard — abort early rather than fill the drive
        free_gb = _shutil.disk_usage(output_dir).free / (1024 ** 3)
        if free_gb < 20:
            raise OSError(
                f"Disk nearly full ({free_gb:.1f} GB free) — aborting to protect data. "
                "Free space and re-run; already-appended tokens are safely in train.bin."
            )

    for source_name in source_names:
        source_path = os.path.join(data_dir, source_name)
        if not os.path.exists(source_path):
            print(f"SKIP {source_name}: not found at {source_path}")
            continue

        text_field = _SOURCE_FIELD.get(source_name, "text")
        print(f"Tokenizing {source_name} (field={text_field})...")
        ds = load_from_disk(source_path)
        source_count = 0

        for i, item in enumerate(ds):
            text = item.get(text_field, "")
            if not text:
                for alt in ["text", "content", "query", "instruction"]:
                    text = item.get(alt, "")
                    if text:
                        break
            if not text:
                continue

            toks = tokenizer.encode_ordinary(text)
            toks.append(tokenizer.eos_token_id)
            token_buf.extend(toks)
            source_count += len(toks)

            if len(token_buf) >= flush_every:
                _flush()

            if (i + 1) % 200_000 == 0:
                free_gb = _shutil.disk_usage(output_dir).free / (1024 ** 3)
                print(f"  {source_name}: {i+1} docs, {source_count:,} tokens  (disk free: {free_gb:.0f} GB)")

        _flush()
        print(f"  {source_name}: {source_count:,} tokens total")

    total_new = total_new_train + total_new_val
    print(f"\nNew tokens appended: {total_new:,}  (train: {total_new_train:,}  val: {total_new_val:,})")
    if total_new == 0:
        print("Nothing was appended.")
        return

    # Report final sizes
    final_train = os.path.getsize(train_path) // 4
    final_val   = os.path.getsize(val_path) // 4 if os.path.exists(val_path) else 0
    print(f"\nDone!")
    print(f"  train.bin: {final_train:,} tokens  ({final_train*4/1e9:.1f} GB)")
    print(f"  val.bin:   {final_val:,} tokens  ({final_val*4/1e9:.1f} GB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tokenize", "append", "phase", "test"],
                        default="tokenize")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--fim-rate", type=float, default=0.0,
                        help="FIM augmentation probability for code sources (0.0–1.0)")
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated list of source names to append (used with --stage append). "
             "E.g. c4_en,cc_news,code_java,code_javascript",
    )
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase name for --stage phase (e.g. phase1, phase2, phase3)")
    args = parser.parse_args()

    if args.stage == "tokenize":
        tokenize_pretrain_data(max_tokens=args.max_tokens, fim_rate=args.fim_rate)
    elif args.stage == "append":
        if not args.sources:
            print("ERROR: --sources required with --stage append")
            raise SystemExit(1)
        append_new_sources(sources_csv=args.sources)
    elif args.stage == "phase":
        if not args.phase:
            print("ERROR: --phase required with --stage phase")
            raise SystemExit(1)
        tokenize_phase(phase_name=args.phase, max_tokens=args.max_tokens,
                       fim_rate=args.fim_rate)
    elif args.stage == "test":
        tokenizer = get_tokenizer()
        print(f"Vocab size: {tokenizer.vocab_size}")
        tokens = tokenizer.encode("Hello, world! def foo():\n    return 42")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {tokenizer.decode(tokens)}")
