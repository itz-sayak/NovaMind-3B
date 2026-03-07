"""
Supervised Fine-Tuning (SFT) script for NovaMind-3B.

Loads a pretrained checkpoint and fine-tunes on instruction-following data
(code, math, and general chat). Uses proper loss masking on instruction tokens.
"""
import os
import sys
import time
import math
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.model_config import NovaMind3BConfig
from configs.train_config import SFTConfig
from model.transformer import NovaMind3B
from data.dataset import SFTDataset


def get_lr(step, total_steps, config):
    """Cosine decay with warmup for SFT."""
    warmup_steps = min(100, total_steps // 10)
    
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    
    decay_steps = total_steps - warmup_steps
    current_step = step - warmup_steps
    decay_ratio = current_step / max(1, decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train_sft(args):
    sft_config = SFTConfig()
    model_config = NovaMind3BConfig()
    model_config.mtp_depth = 0  # No MTP during SFT
    # Apply context extension overrides from SFTConfig.
    model_config.max_seq_len = sft_config.max_seq_len
    model_config.rope_scale_factor = sft_config.rope_scale_factor
    
    if args.batch_size:
        sft_config.batch_size = args.batch_size
    if args.lr:
        sft_config.learning_rate = args.lr
    
    device = sft_config.device
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("NovaMind-3B Supervised Fine-Tuning")
    print("=" * 60)
    
    # Load model
    print("\nInitializing model...")
    model = NovaMind3B(model_config)
    
    # Load pretrained weights
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        # Remove MTP weights if present (they're not used in SFT)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mtp_module")}
        model.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: No pretrained weights loaded. Training from scratch.")
    
    model = model.to(device)
    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,}")
    
    # Dataset
    print(f"\nLoading SFT data from {sft_config.data_dir}...")
    try:
        train_dataset = SFTDataset(
            sft_config.data_dir, max_len=model_config.max_seq_len, split="train"
        )
        val_dataset = SFTDataset(
            sft_config.data_dir, max_len=model_config.max_seq_len, split="val"
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run: python data/download.py --stage sft")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=sft_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=sft_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    # Calculate total steps
    steps_per_epoch = len(train_loader) // sft_config.gradient_accumulation_steps
    total_steps = sft_config.num_epochs * steps_per_epoch
    if sft_config.max_steps > 0:
        total_steps = min(total_steps, sft_config.max_steps)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    
    # Optimizer — simple AdamW for SFT (no Muon)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=sft_config.weight_decay,
    )
    
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
    
    # Training loop
    print("\nStarting SFT training...")
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(sft_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{sft_config.num_epochs}")
        train_iter = iter(train_loader)
        running_loss = 0.0
        t0 = time.time()
        
        for local_step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            with ctx:
                result = model(input_ids, targets=labels)
                # Apply loss mask (only compute loss on response tokens)
                logits = result["logits"]
                # Recompute loss with mask
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = loss_mask[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
            
            scaled_loss = loss / sft_config.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += loss.item() / sft_config.gradient_accumulation_steps
            
            if (local_step + 1) % sft_config.gradient_accumulation_steps == 0:
                if sft_config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)
                
                # Update LR
                lr = get_lr(global_step, total_steps, sft_config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                if global_step % sft_config.log_interval == 0:
                    dt = time.time() - t0
                    print(
                        f"  step {global_step:>5d}/{total_steps} | "
                        f"loss {running_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"{dt:.1f}s"
                    )
                    running_loss = 0.0
                    t0 = time.time()
                
                if global_step >= total_steps:
                    break
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 50:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                
                with ctx:
                    result = model(input_ids, targets=labels)
                    logits = result["logits"]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    shift_mask = loss_mask[:, 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )
                    loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
                
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        print(f"  Epoch {epoch+1} val_loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(sft_config.output_dir, "sft_best.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  Saved best model to {ckpt_path}")
        
        model.train()
        
        if global_step >= total_steps:
            break
    
    # Final save
    final_path = os.path.join(sft_config.output_dir, "sft_final.pt")
    torch.save({"model": model.state_dict(), "step": global_step}, final_path)
    print(f"\nSFT complete! Final model saved to {final_path}")


# Need F for loss computation
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaMind-3B SFT")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    
    train_sft(args)
