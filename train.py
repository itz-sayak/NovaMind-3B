"""
Pretraining script for NovaMind-3B.

Features:
  - Mixed precision (bfloat16) training
  - Gradient checkpointing for memory efficiency 
  - Muon optimizer for hidden layers + AdamW for rest
  - Cosine learning rate schedule with warmup
  - Multi-Token Prediction auxiliary objective
  - Auxiliary-loss-free expert load balancing
  - Wandb logging (optional)
  - Checkpoint saving and resuming
"""
import os
import sys
import time
import math
import json
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")  # use TF32 on Ampere+

# ── Compile / memory config ──────────────────────────────────────────────────
# The inductor pad-mm pass benchmarks padded matmuls on-GPU, temporarily
# allocating ~3 GiB.  On 24 GB cards with a 1B model it OOMs.
# Patch should_pad_common to always return False so the pass is skipped.
import torch._inductor.fx_passes.pad_mm as _pad_mm_mod
_pad_mm_mod.should_pad_common = lambda *_a, **_kw: False
# Reduce fragmentation with expandable CUDA allocator segments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Pre-parse FLA_DISABLE before model imports ────────────────────────────
# gated_delta_net.py checks FLA_DISABLE at import time, so we must set it
# before importing the model. Parse --smoke-test/--no-fla minimally here.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--smoke-test", action="store_true")
_pre.add_argument("--no-fla", action="store_true")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.no_fla or _pre_args.smoke_test:
    os.environ.setdefault("FLA_DISABLE", "1")
del _pre, _pre_args

from configs.model_config import NovaMind3BConfig
from configs.train_config import PretrainConfig
from model.transformer import NovaMind3B
from data.dataset import PretrainDataset, StreamingPretrainDataset
from optim.muon import create_optimizer

try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False


# ── Distributed helpers ───────────────────────────────────────────────────────
def init_distributed():
    """Initialise process group if launched with torchrun, else single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def is_main(rank):
    return rank == 0


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_lr(step, config):
    """WSD learning-rate schedule: warmup -> constant -> cosine decay to 0.

    Phase 1 (warmup):  0 -> warmup_steps               linear ramp
    Phase 2 (stable):  warmup_steps -> decay_start      constant at peak LR
    Phase 3 (decay):   decay_start -> max_steps          cosine to min_lr
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_start = int(config.max_steps * (1.0 - config.decay_fraction))

    if step < decay_start:
        return config.learning_rate

    decay_len = config.max_steps - decay_start
    progress = (step - decay_start) / max(1, decay_len)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def get_muon_lr(step, config):
    """WSD schedule for Muon parameters (mirrors AdamW schedule shape)."""
    base_lr = config.muon_lr
    min_lr = 0.0  # decay to 0

    if step < config.warmup_steps:
        return base_lr * (step + 1) / config.warmup_steps

    decay_start = int(config.max_steps * (1.0 - config.decay_fraction))

    if step < decay_start:
        return base_lr

    decay_len = config.max_steps - decay_start
    progress = (step - decay_start) / max(1, decay_len)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def get_mtp_weight(step, config):
    """MTP loss weight annealing: lambda drops at decay onset."""
    if not config.use_mtp:
        return 0.0
    decay_start = int(config.max_steps * (1.0 - config.decay_fraction))
    if step < decay_start:
        return config.mtp_loss_weight
    return config.mtp_loss_weight_final


def get_grad_accum(step, config):
    """Batch-size warmup: ramp gradient accumulation from initial -> full."""
    if step >= config.grad_accum_warmup_steps:
        return config.gradient_accumulation_steps
    frac = step / max(1, config.grad_accum_warmup_steps)
    val = config.grad_accum_initial + frac * (
        config.gradient_accumulation_steps - config.grad_accum_initial
    )
    return max(config.grad_accum_initial, int(val))


class EMA:
    """Exponential Moving Average of model parameters (CPU-resident)."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        raw = model.module if isinstance(model, DDP) else model
        for name, param in raw.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().cpu().clone()

    @torch.no_grad()
    def update(self, model):
        raw = model.module if isinstance(model, DDP) else model
        for name, param in raw.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float().cpu(), alpha=1.0 - self.decay
                )

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)

    def apply_to(self, model):
        """Copy EMA weights into model (e.g. for evaluation)."""
        raw = model.module if isinstance(model, DDP) else model
        for name, param in raw.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device, param.dtype))


@torch.no_grad()
def evaluate(model, val_dataloader, config, ctx, rank, max_batches=50):
    """Run evaluation on validation set (all ranks, average on rank 0)."""
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    losses = []

    for i, (x, y) in enumerate(val_dataloader):
        if i >= max_batches:
            break
        x, y = x.to(config.device), y.to(config.device)
        with ctx:
            result = (model.module if isinstance(model, DDP) else model)(x, targets=y)
        losses.append(result["loss"].item())

    raw.train()
    return sum(losses) / len(losses) if losses else float("inf")


def save_checkpoint(model, optimizer, step, config, loss, path, ema=None):
    """Save training checkpoint (call only from rank 0)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": {
            "model": vars(NovaMind3BConfig()),
            "train": vars(config),
        },
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)
    # Also save as latest
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, ema=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    if ema is not None and "ema" in checkpoint:
        try:
            ema.load_state_dict(checkpoint["ema"])
        except Exception as e:
            print(f"Warning: Could not load EMA state: {e}")
    return checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))


# ── W&B helpers ──────────────────────────────────────────────────────────────
def init_wandb(args, model_config, train_config, world_size):
    """Initialise Weights & Biases run on rank 0 only."""
    if not _wandb_available:
        print("Warning: wandb not installed. Skipping W&B logging.")
        return False
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity or None,
        config={
            # Model
            "vocab_size":        model_config.vocab_size,
            "hidden_dim":        model_config.hidden_dim,
            "num_layers":        model_config.num_layers,
            "n_heads":           model_config.n_heads,
            "max_seq_len":       model_config.max_seq_len,
            "rope_base":         model_config.rope_base,
            "n_routed_experts":  model_config.n_routed_experts,
            "n_activated_experts": model_config.n_activated_experts,
            "mtp_depth":         model_config.mtp_depth,
            # Training
            "batch_size_per_gpu":         train_config.batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "world_size":                 world_size,
            "effective_batch_tokens":     (train_config.batch_size * world_size *
                                           train_config.gradient_accumulation_steps *
                                           model_config.max_seq_len),
            "max_steps":         train_config.max_steps,
            "warmup_steps":      train_config.warmup_steps,
            "learning_rate":     train_config.learning_rate,
            "min_lr":            train_config.min_lr,
            "muon_lr":           train_config.muon_lr,
            "weight_decay":      train_config.weight_decay,
            "grad_clip":         train_config.grad_clip,
            "dtype":             train_config.dtype,
            "compile":           train_config.compile,
            "gradient_checkpointing": train_config.gradient_checkpointing,
            "mtp_loss_weight":   train_config.mtp_loss_weight,
        },
        resume="allow",
    )
    print(f"  W&B run: {wandb.run.get_url()}")
    return True


def log_wandb(metrics: dict, step: int):
    """Safe wandb log — no-ops if wandb is not active."""
    if _wandb_available and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    if _wandb_available and wandb.run is not None:
        wandb.finish()


def train(args):
    # ── Distributed init ──────────────────────────────────────────────────────
    rank, world_size, local_rank = init_distributed()
    main = is_main(rank)

    # Configuration
    model_config = NovaMind3BConfig()
    train_config = PretrainConfig()

    # Override with command line args
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.grad_accum:
        train_config.gradient_accumulation_steps = args.grad_accum
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.seq_len:
        model_config.max_seq_len = args.seq_len
    if args.no_mtp:
        model_config.mtp_depth = 0
        train_config.use_mtp = False
    if args.no_compile:
        train_config.compile = False
    if args.data_dir:
        train_config.data_dir = args.data_dir
    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.smoke_test:
        # Tiny model (~50M params) for local validation — does not fit the 3B config on 24 GB
        model_config.hidden_dim = 512
        model_config.num_layers = 4
        model_config.num_dense_layers = 4
        model_config.gdn_num_heads = 2
        model_config.gdn_head_dim = 64
        model_config.n_heads = 4
        model_config.d_head = 128
        model_config.d_kv_comp = 128
        model_config.d_q_comp = 256
        model_config.d_rope = 64
        model_config.dense_intermediate = 1024
        model_config.hybrid_attention_layers = [3]
        model_config.mtp_depth = 0
        train_config.use_mtp = False
        train_config.compile = False
        train_config.num_workers = 0   # avoid fork+CUDA issues on constrained hardware
        if not args.seq_len:
            model_config.max_seq_len = 256
        if not args.max_steps:
            train_config.max_steps = 5
        if not args.batch_size:
            train_config.batch_size = 1
        if not args.grad_accum:
            train_config.gradient_accumulation_steps = 1
        if main:
            print("[smoke-test] Using tiny model config (~50M params)")

    # ── W&B init (rank 0 only) ────────────────────────────────────────────────
    wandb_enabled = False
    if main and args.wandb:
        wandb_enabled = init_wandb(args, model_config, train_config, world_size)

    # In DDP each GPU gets its own device
    if world_size > 1:
        train_config.device = f"cuda:{local_rank}"

    device = train_config.device
    dtype  = torch.bfloat16 if train_config.dtype == "bfloat16" else torch.float16

    if main:
        print("=" * 60)
        print(f"NovaMind-3B Pretraining  (world_size={world_size})")
        print("=" * 60)
    
    # Model
    if main:
        print("\nInitializing model...")
    model = NovaMind3B(model_config)
    param_counts = model.count_parameters()
    if main:
        print(f"Total parameters: {param_counts['total']:,} ({param_counts['total']/1e9:.3f}B)")
        print(f"Trainable parameters: {param_counts['trainable']:,}")

        config_counts = model_config.count_parameters()
        print(f"\nParameter breakdown (config):")
        for k, v in config_counts.items():
            if isinstance(v, int):
                print(f"  {k}: {v:,}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6f}")

    model = model.to(device=device, dtype=torch.bfloat16)

    # Compile before DDP wrap (torch.compile + DDP is supported in PyTorch 2.x)
    if train_config.compile and hasattr(torch, "compile"):
        if main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if main:
            print(f"  DDP enabled across {world_size} GPUs")
    
    # Optimizer (built on raw model, before DDP)
    if main:
        print("\nSetting up Muon + AdamW optimizer...")
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = create_optimizer(raw_model, train_config)

    # EMA (Exponential Moving Average on CPU for early decay estimation)
    ema = None
    if train_config.ema_enabled:
        ema = EMA(model, decay=train_config.ema_decay)
        if main:
            print(f"  EMA enabled (decay={train_config.ema_decay})")

    # Data
    if main:
        print(f"\nLoading data from {train_config.data_dir}...")
    try:
        train_dataset = StreamingPretrainDataset(
            train_config.data_dir,
            seq_len=model_config.max_seq_len,
            split="train",
            shuffle_buffer=train_config.shuffle_buffer,
            world_size=world_size,
            rank=rank,
        )
        val_dataset = StreamingPretrainDataset(
            train_config.data_dir,
            seq_len=model_config.max_seq_len,
            split="val",
            shuffle_buffer=1,      # val: no shuffle needed
            world_size=world_size,
            rank=rank,
        )
    except FileNotFoundError as e:
        if main:
            print(f"\nERROR: {e}")
            print("Please run data preparation first:")
            print("  python data/download.py --stage pretrain")
            print("  python data/dataset.py --stage tokenize")
        destroy_distributed()
        return

    # StreamingPretrainDataset handles sharding/shuffling internally — no sampler needed
    train_sampler = None
    val_sampler   = None

    nw = train_config.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        num_workers=nw,
        pin_memory=(nw > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        num_workers=max(0, nw - 2),
        pin_memory=(nw > 0),
        drop_last=True,
    )
    
    # Mixed precision context
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == torch.float16))
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_path = args.resume
        if os.path.exists(resume_path):
            if main:
                print(f"Resuming from {resume_path}")
            raw_model = model.module if isinstance(model, DDP) else model
            start_step, _ = load_checkpoint(resume_path, raw_model, optimizer, ema=ema)
            if main:
                print(f"Resumed at step {start_step}")

    # Training loop
    if main:
        print(f"\nStarting training from step {start_step}")
        print(f"  Batch size per GPU: {train_config.batch_size}")
        print(f"  GPUs: {world_size}")
        print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps} "
              f"(warmup from {train_config.grad_accum_initial} over "
              f"{train_config.grad_accum_warmup_steps} steps)")
        eff_bs = train_config.batch_size * world_size * train_config.gradient_accumulation_steps
        print(f"  Effective batch size (peak): {eff_bs}")
        print(f"  Sequence length: {model_config.max_seq_len}")
        print(f"  Max steps: {train_config.max_steps}")
        max_tok_per_step = eff_bs * model_config.max_seq_len
        print(f"  Tokens/step (peak): {max_tok_per_step:,}")
        print(f"  Total tokens (approx): ~{max_tok_per_step * train_config.max_steps / 1e9:.1f}B")
        decay_start = int(train_config.max_steps * (1.0 - train_config.decay_fraction))
        print(f"  LR schedule: WSD (constant until step {decay_start}, "
              f"then cosine decay over final {train_config.decay_fraction:.0%})")
        print(f"  MTP lambda: {train_config.mtp_loss_weight} -> "
              f"{train_config.mtp_loss_weight_final} at step {decay_start}")
        if train_config.data_phases:
            print(f"  Data phases: {len(train_config.data_phases)}")
            for i, (frac, d) in enumerate(train_config.data_phases):
                print(f"    Phase {i+1}: step {int(frac * train_config.max_steps)} -> {d}")
        print()

    # Phase tracking for multi-stage data rebalancing
    current_phase_idx = 0
    if train_config.data_phases:
        for i, (frac, _) in enumerate(train_config.data_phases):
            if start_step >= frac * train_config.max_steps:
                current_phase_idx = i

    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    running_loss = 0.0
    running_mtp_loss = 0.0
    running_grad_norm = 0.0
    total_tokens = 0  # cumulative counter (accounts for batch-size warmup)
    t0 = time.time()

    for step in range(start_step, train_config.max_steps):
        # ── Data phase check ───────────────────────────────────────────
        if train_config.data_phases:
            target_phase = 0
            for i, (frac, _) in enumerate(train_config.data_phases):
                if step >= frac * train_config.max_steps:
                    target_phase = i
            if target_phase != current_phase_idx:
                current_phase_idx = target_phase
                _, new_dir = train_config.data_phases[current_phase_idx]
                if main:
                    print(f"\n{'='*60}")
                    print(f"  DATA PHASE {current_phase_idx+1}: switching to {new_dir}")
                    print(f"{'='*60}\n")
                try:
                    train_dataset = StreamingPretrainDataset(
                        new_dir, seq_len=model_config.max_seq_len, split="train",
                        shuffle_buffer=train_config.shuffle_buffer,
                        world_size=world_size, rank=rank,
                    )
                    val_dataset = StreamingPretrainDataset(
                        new_dir, seq_len=model_config.max_seq_len, split="val",
                        shuffle_buffer=1, world_size=world_size, rank=rank,
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=train_config.batch_size,
                        num_workers=nw, pin_memory=(nw > 0), drop_last=True,
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=train_config.batch_size,
                        num_workers=max(0, nw - 2), pin_memory=(nw > 0), drop_last=True,
                    )
                    train_iter = iter(train_loader)
                except FileNotFoundError as e:
                    if main:
                        print(f"  Warning: Phase data not found ({e}), keeping current data")

        # ── Schedule updates ───────────────────────────────────────────
        lr = get_lr(step, train_config)
        muon_lr = get_muon_lr(step, train_config)
        mtp_weight = get_mtp_weight(step, train_config)
        grad_accum = get_grad_accum(step, train_config)
        for param_group in optimizer.param_groups:
            if param_group.get("use_muon", False):
                param_group["lr"] = muon_lr
            else:
                param_group["lr"] = lr
        
        # Gradient accumulation loop
        # Use DDP no_sync() on all but the last micro-step to avoid redundant
        # all-reduce communication during accumulation.
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Only sync gradients on the last micro-step
            sync_ctx = nullcontext() if (not isinstance(model, DDP) or micro_step == grad_accum - 1) \
                       else model.no_sync()

            with sync_ctx:
                with ctx:
                    raw = model.module if isinstance(model, DDP) else model
                    result = raw(x, targets=y)
                    loss     = result["loss"]
                    mtp_loss = result["mtp_loss"]

                    total_loss = loss
                    if train_config.use_mtp and mtp_loss.item() > 0:
                        total_loss = total_loss + mtp_weight * mtp_loss

                    scaled_loss = total_loss / grad_accum

                scaler.scale(scaled_loss).backward()

            running_loss     += loss.item() / grad_accum
            running_mtp_loss += mtp_loss.item() / grad_accum

        # Gradient clipping + grad norm tracking
        grad_norm = 0.0
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            ).item()

        running_grad_norm += grad_norm
        
        scaler.step(optimizer)
        scaler.update()

        # EMA update (async to CPU)
        if ema is not None:
            ema.update(model)

        # Track actual tokens processed (accounts for batch-size warmup)
        total_tokens += train_config.batch_size * world_size * grad_accum * model_config.max_seq_len

        # Update expert biases (auxiliary-loss-free balancing)
        if result.get("expert_counts"):
            raw_model = model.module if isinstance(model, DDP) else model
            for layer_idx, expert_counts in result["expert_counts"]:
                layer = raw_model.layers[layer_idx]
                if hasattr(layer.ffn, "router"):
                    layer.ffn.router.update_expert_bias(expert_counts)

        # Logging (rank 0 only)
        if main and (step + 1) % train_config.log_interval == 0:
            dt = time.time() - t0
            tokens_processed = total_tokens
            step_tokens = train_config.batch_size * world_size * grad_accum * model_config.max_seq_len
            tokens_per_sec = step_tokens * train_config.log_interval / dt
            # Divide by log_interval to get per-step averages
            avg_loss      = running_loss     / train_config.log_interval
            avg_mtp_loss  = running_mtp_loss / train_config.log_interval
            avg_grad_norm = running_grad_norm / train_config.log_interval
            ppl = math.exp(min(avg_loss, 20))
            gpu_mem_gb = [
                torch.cuda.memory_allocated(i) / 1e9
                for i in range(torch.cuda.device_count())
            ]
            gpu_reserved_gb = [
                torch.cuda.memory_reserved(i) / 1e9
                for i in range(torch.cuda.device_count())
            ]

            print(
                f"step {step+1:>6d}/{train_config.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"ppl {ppl:.2f} | "
                f"mtp {avg_mtp_loss:.4f} (\u03bb={mtp_weight:.2f}) | "
                f"gnorm {avg_grad_norm:.3f} | "
                f"lr {lr:.2e} | "
                f"muon_lr {muon_lr:.2e} | "
                f"tok/s {tokens_per_sec:.0f} | "
                f"ga {grad_accum} | "
                f"tokens {tokens_processed/1e9:.2f}B"
            )

            if wandb_enabled:
                log_wandb({
                    "train/loss":        avg_loss,
                    "train/ppl":         ppl,
                    "train/mtp_loss":    avg_mtp_loss,
                    "train/mtp_weight":  mtp_weight,
                    "train/grad_norm":   avg_grad_norm,
                    "optimizer/lr":      lr,
                    "optimizer/muon_lr": muon_lr,
                    "optimizer/grad_accum": grad_accum,
                    "perf/tok_per_sec":  tokens_per_sec,
                    "perf/tokens_seen_B": tokens_processed / 1e9,
                    **{f"gpu/{i}/mem_alloc_gb":    gpu_mem_gb[i]    for i in range(len(gpu_mem_gb))},
                    **{f"gpu/{i}/mem_reserved_gb": gpu_reserved_gb[i] for i in range(len(gpu_reserved_gb))},
                }, step=step + 1)

            running_loss = 0.0
            running_mtp_loss = 0.0
            running_grad_norm = 0.0
            t0 = time.time()
        
        # Evaluation
        if (step + 1) % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_config, ctx, rank)
            if main:
                val_ppl = math.exp(min(val_loss, 20))
                tokens_seen_B = total_tokens / 1e9
                print(f"  >> val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}")
                if wandb_enabled:
                    log_wandb({
                        "val/loss":       val_loss,
                        "val/ppl":        val_ppl,
                        "val/best_loss":  min(val_loss, best_val_loss),
                        "perf/tokens_seen_B": tokens_seen_B,
                    }, step=step + 1)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, step + 1, train_config, val_loss,
                        os.path.join(train_config.output_dir, "best.pt"),
                        ema=ema,
                    )

        # Periodic save (rank 0 only)
        if main and (step + 1) % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1, train_config, running_loss,
                os.path.join(train_config.output_dir, f"step_{step+1}.pt"),
                ema=ema,
            )

    # Final save
    if main:
        save_checkpoint(
            model, optimizer, train_config.max_steps, train_config, running_loss,
            os.path.join(train_config.output_dir, "final.pt"),
            ema=ema,
        )
        print("\nTraining complete!")
        finish_wandb()

    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaMind-3B Pretraining")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-fla", action="store_true",
                        help="Disable FLA Triton kernels, use pure-PyTorch GDN fallback")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Shrink model to tiny size for local smoke testing")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override PretrainConfig.data_dir (path containing train.bin/val.bin)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override PretrainConfig.output_dir (checkpoint save path)")
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="novamind-3b-pretrain")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team (leave blank for personal account)")
    args = parser.parse_args()

    train(args)
