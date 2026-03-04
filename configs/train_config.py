"""
Training configuration for NovaMind-3B (2× NVIDIA L40S).
"""
from dataclasses import dataclass, field


@dataclass
class PretrainConfig:
    """Stage 1: Pretraining on code + math + general text."""
    # Data
    data_dir: str = "/mnt/zone/A/datasets/pretrain"
    max_seq_len: int = 8192
    
    # Training
    # 2 GPUs × batch=1 × grad_accum=16 × seq=8192 = 262,144 tokens/step
    # 400k steps × 262k = ~105B tokens (Chinchilla-optimal for 3B is 60B;
    # 105B intentionally overtrained for inference-efficiency: Phi/TinyLlama style)
    batch_size: int = 1             # micro batch per GPU (safe for 3B @ 48GB)
    gradient_accumulation_steps: int = 16
    max_steps: int = 400_000
    warmup_steps: int = 2000
    
    # Optimizer (Muon + AdamW)
    # 3B scale: keep 2.2e-4 (LLaMA-3.2-3B uses 3e-4; 2.2e-4 is conservative/stable)
    learning_rate: float = 2.2e-4
    min_lr: float = 0.0             # WSD decays to 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Muon-specific
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    
    # Schedule (WSD = Warmup-Stable-Decay)
    lr_schedule: str = "wsd"
    decay_fraction: float = 0.15    # final 15% of training is cosine decay to 0
    
    # IO
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 5_000
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-3b/pretrain"
    
    # Batch-size warmup (grad_accum ramps from initial -> full)
    grad_accum_initial: int = 4
    grad_accum_warmup_steps: int = 5000

    # EMA (stored on CPU)
    ema_enabled: bool = True
    ema_decay: float = 0.9995

    # Multi-stage data phases
    data_phases: list = field(default_factory=list)

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    
    # Multi-GPU — set ddp=True and launch with torchrun --nproc_per_node=2
    ddp: bool = True
    
    # MTP
    use_mtp: bool = True
    mtp_loss_weight: float = 0.3
    mtp_loss_weight_final: float = 0.1


@dataclass
class SFTConfig:
    """Stage 2: Supervised Fine-Tuning on instruction data."""
    data_dir: str = "/mnt/zone/A/datasets/sft"
    max_seq_len: int = 2048
    
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 2
    max_steps: int = -1
    warmup_steps: int = 100
    
    learning_rate: float = 5e-6
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    lr_schedule: str = "cosine"
    
    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-3b/sft"
    
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    
    pretrained_ckpt: str = "/mnt/zone/A/checkpoints/novamind-3b/pretrain/latest.pt"


@dataclass
class DPOConfig:
    """Stage 3: Direct Preference Optimization."""
    data_dir: str = "/mnt/zone/A/datasets/dpo"
    max_seq_len: int = 2048
    
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 5000
    warmup_steps: int = 100
    
    learning_rate: float = 5e-7
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    beta: float = 0.1  # KL penalty coefficient
    
    lr_schedule: str = "cosine"
    
    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "/mnt/zone/A/checkpoints/novamind-3b/dpo"
    
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True
    
    sft_ckpt: str = "/mnt/zone/A/checkpoints/novamind-3b/sft/latest.pt"
