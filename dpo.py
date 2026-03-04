"""
Direct Preference Optimization (DPO) script for NovaMind-3B.

Implements the DPO algorithm from "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model" (Rafailov et al., 2023).

Takes an SFT checkpoint and further aligns it using preference data.
"""
import os
import sys
import time
import math
import copy
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.model_config import NovaMind3BConfig
from configs.train_config import DPOConfig
from model.transformer import NovaMind3B
from data.dataset import DPODataset


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    Compute DPO loss.
    
    loss = -log(sigmoid(beta * (log(pi(y_w|x)/ref(y_w|x)) - log(pi(y_l|x)/ref(y_l|x)))))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    # Metrics
    with torch.no_grad():
        chosen_acc = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
    
    return loss, chosen_acc, reward_margin


def get_log_probs(model, input_ids, labels, loss_mask, ctx):
    """Get per-sequence sum of log probabilities."""
    with ctx:
        result = model(input_ids)
        logits = result["logits"]
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()
    
    # Per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Masked sum per sequence
    per_seq_logps = (per_token_logps * shift_mask).sum(dim=-1)
    
    return per_seq_logps


def train_dpo(args):
    model_config = NovaMind3BConfig()
    model_config.mtp_depth = 0
    dpo_config = DPOConfig()
    
    if args.batch_size:
        dpo_config.batch_size = args.batch_size
    if args.beta:
        dpo_config.beta = args.beta
    
    device = dpo_config.device
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("NovaMind-3B DPO Training")
    print("=" * 60)
    
    # Initialize models
    print("\nInitializing policy and reference models...")
    policy_model = NovaMind3B(model_config)
    ref_model = NovaMind3B(model_config)
    
    # Load SFT checkpoint
    if args.sft_checkpoint:
        print(f"Loading SFT checkpoint from {args.sft_checkpoint}")
        ckpt = torch.load(args.sft_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        policy_model.load_state_dict(state_dict, strict=False)
        ref_model.load_state_dict(state_dict, strict=False)
    else:
        print("ERROR: SFT checkpoint required for DPO training.")
        return
    
    policy_model = policy_model.to(device)
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # Freeze reference model
    for p in ref_model.parameters():
        p.requires_grad = False
    
    print(f"Parameters: {policy_model.count_parameters()['total']:,}")
    
    # Dataset
    print(f"\nLoading DPO data from {dpo_config.data_dir}...")
    try:
        train_dataset = DPODataset(
            dpo_config.data_dir, max_len=model_config.max_seq_len, split="train"
        )
        val_dataset = DPODataset(
            dpo_config.data_dir, max_len=model_config.max_seq_len, split="val"
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run: python data/download.py --stage dpo")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=dpo_config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=dpo_config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True, drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=dpo_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=dpo_config.weight_decay,
    )
    
    total_steps = dpo_config.max_steps if dpo_config.max_steps > 0 else len(train_loader)
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
    
    # Training loop
    print(f"\nStarting DPO training for {total_steps} steps")
    print(f"  Beta: {dpo_config.beta}")
    print(f"  Batch size: {dpo_config.batch_size}")
    print()
    
    policy_model.train()
    global_step = 0
    best_val_acc = 0.0
    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()
    
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
            
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            
            # Get policy log probs
            policy_chosen_logps = get_log_probs(
                policy_model, chosen_ids, chosen_labels, chosen_mask, ctx
            )
            policy_rejected_logps = get_log_probs(
                policy_model, rejected_ids, rejected_labels, rejected_mask, ctx
            )
            
            # Get reference log probs (no grad)
            with torch.no_grad():
                ref_chosen_logps = get_log_probs(
                    ref_model, chosen_ids, chosen_labels, chosen_mask, ctx
                )
                ref_rejected_logps = get_log_probs(
                    ref_model, rejected_ids, rejected_labels, rejected_mask, ctx
                )
            
            # DPO loss
            loss, acc, margin = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=dpo_config.beta,
            )
            
            loss.backward()
            
            if dpo_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), dpo_config.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            
            running_loss += loss.item()
            running_acc += acc.item()
            
            # Logging
            if global_step % dpo_config.log_interval == 0:
                dt = time.time() - t0
                avg_loss = running_loss / dpo_config.log_interval
                avg_acc = running_acc / dpo_config.log_interval
                print(
                    f"step {global_step:>5d}/{total_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"acc {avg_acc:.3f} | "
                    f"margin {margin.item():.3f} | "
                    f"{dt:.1f}s"
                )
                running_loss = 0.0
                running_acc = 0.0
                t0 = time.time()
            
            # Eval
            if global_step % dpo_config.eval_interval == 0:
                policy_model.eval()
                val_accs = []
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        if i >= 30:
                            break
                        chosen_ids = batch["chosen_ids"].to(device)
                        chosen_labels = batch["chosen_labels"].to(device)
                        chosen_mask = batch["chosen_mask"].to(device)
                        rejected_ids = batch["rejected_ids"].to(device)
                        rejected_labels = batch["rejected_labels"].to(device)
                        rejected_mask = batch["rejected_mask"].to(device)
                        
                        pc = get_log_probs(policy_model, chosen_ids, chosen_labels, chosen_mask, ctx)
                        pr = get_log_probs(policy_model, rejected_ids, rejected_labels, rejected_mask, ctx)
                        rc = get_log_probs(ref_model, chosen_ids, chosen_labels, chosen_mask, ctx)
                        rr = get_log_probs(ref_model, rejected_ids, rejected_labels, rejected_mask, ctx)
                        
                        _, vacc, _ = dpo_loss(pc, pr, rc, rr, dpo_config.beta)
                        val_accs.append(vacc.item())
                
                val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
                print(f"  >> val_acc: {val_acc:.3f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_path = os.path.join(dpo_config.output_dir, "dpo_best.pt")
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save({
                        "model": policy_model.state_dict(),
                        "step": global_step,
                        "val_acc": val_acc,
                    }, ckpt_path)
                    print(f"  Saved best model (acc={val_acc:.3f})")
                
                policy_model.train()
    
    # Final save
    final_path = os.path.join(dpo_config.output_dir, "dpo_final.pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({"model": policy_model.state_dict(), "step": global_step}, final_path)
    print(f"\nDPO training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NovaMind-3B DPO")
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()
    
    train_dpo(args)
