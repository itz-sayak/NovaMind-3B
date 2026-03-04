"""
Dense 3B Reasoning Model Configuration.

Architecture: MLA + Dense SwiGLU FFN + RMSNorm + RoPE + MTP
Total Parameters: ~3.0B (all activated per token — no MoE)
Target hardware: 2× NVIDIA L40S (2× 48 GB) via FSDP ZeRO-2

Design rationale (3B scale on 2× L40S):
  - hidden 1536→3072, layers 24→26, heads 16→24 (d_head 96→128)
  - FFN intermediate 6144→8192 (≈2.67× hidden; SwiGLU ≈4× effective)
  - Keeps all 1B design choices: dense, MLA, MTP, RoPE-500k, cl100k
  - VRAM budget: ~34 GB/GPU with FSDP ZeRO-2 + activation checkpointing
    (weights 3GB + grads 6GB + Adam 12GB + activations ~12GB = ~33GB)
  - Both GPUs run at full utilisation; no ZeRO-3 / offload needed
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NovaMind3BConfig:
    # --- Tokenizer ---
    vocab_size: int = 100352          # cl100k_base padded to multiple of 64

    # --- Transformer core ---
    hidden_dim: int = 3072            # 2× the 1B hidden (1536→3072)
    num_layers: int = 26              # 1B had 24; 26 × 104M ≈ 3.0B total
    max_seq_len: int = 8192
    dropout: float = 0.0

    # --- Multi-head Latent Attention (MLA) ---
    n_heads: int = 24                 # 16→24; d_head=128 (24×128=3072)
    d_head: int = 128                 # 96→128
    d_kv_comp: int = 768              # hidden/4  (384→768)
    d_q_comp: int = 1536              # hidden/2  (768→1536)
    d_rope: int = 64                  # d_head/2  (48→64)

    # --- Feed-Forward Network (ALL layers dense) ---
    dense_intermediate: int = 8192    # ≈2.67× hidden; SwiGLU ≈4× effective
    num_dense_layers: int = 26        # ALL layers are dense

    # --- MoE (DISABLED — all layers dense) ---
    n_shared_experts: int = 0
    shared_expert_intermediate: int = 0
    n_routed_experts: int = 0
    n_activated_experts: int = 0
    expert_intermediate: int = 0

    # --- Auxiliary-loss-free load balancing (inactive with no MoE) ---
    aux_loss_free: bool = False
    bias_update_speed: float = 0.0
    balance_loss_alpha: float = 0.0

    # --- Multi-Token Prediction ---
    mtp_depth: int = 1                # Predict 1 additional token
    mtp_loss_weight: float = 0.3      # λ

    # --- Training ---
    gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True   # Share embedding ↔ output head
    init_std: float = 0.006           # σ for weight init

    # --- Normalization ---
    rms_norm_eps: float = 1e-6
    rope_base: float = 500_000.0      # Long-range RoPE (LLaMA-3 style)

    @property
    def num_moe_layers(self) -> int:
        return max(0, self.num_layers - self.num_dense_layers)

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        d = self.hidden_dim
        n_h = self.n_heads
        d_h = self.d_head
        d_c = self.d_kv_comp
        d_c_prime = self.d_q_comp
        d_h_R = self.d_rope

        # MLA per layer
        mla_params = (
            d * d_c +                    # W_DKV
            d_c * (d_h * n_h) +          # W_UK
            d_c * (d_h * n_h) +          # W_UV
            d * d_h_R +                  # W_KR
            d * d_c_prime +              # W_DQ
            d_c_prime * (d_h * n_h) +    # W_UQ
            d_c_prime * (d_h_R * n_h) +  # W_QR
            (d_h * n_h) * d              # W_O
        )

        # Dense FFN (SwiGLU): gate + up + down
        dense_ffn_params = d * self.dense_intermediate * 3

        # All layers are dense
        layer_params = mla_params + dense_ffn_params
        total_layers = layer_params * self.num_layers

        # MoE layers (0 if fully dense)
        moe_ffn_params = 0
        total_moe = 0
        if self.n_routed_experts > 0 and self.num_moe_layers > 0:
            shared_expert_params = d * self.shared_expert_intermediate * 3 * self.n_shared_experts
            single_expert_params = d * self.expert_intermediate * 3
            router_params = d * self.n_routed_experts
            moe_ffn_params = shared_expert_params + single_expert_params * self.n_routed_experts + router_params
            # Subtract dense FFN, add MoE FFN for MoE layers
            total_moe = self.num_moe_layers * (moe_ffn_params - dense_ffn_params)

        # RMSNorm (2 per layer + 1 final)
        rmsnorm_params = (self.num_layers * 2 + 1) * d

        # Embedding
        embedding_params = self.vocab_size * d

        # MTP module
        mtp_params = 0
        if self.mtp_depth > 0:
            mtp_projection = 2 * d * d
            mtp_block = mla_params + dense_ffn_params
            mtp_rmsnorm = 3 * d
            mtp_params = (mtp_projection + mtp_block + mtp_rmsnorm) * self.mtp_depth

        total = embedding_params + total_layers + total_moe + rmsnorm_params + mtp_params

        return {
            "embedding": embedding_params,
            "transformer_layers": total_layers,
            "moe_extra (0 if dense)": total_moe,
            "rmsnorm": rmsnorm_params,
            "mtp": mtp_params,
            "total": total,
            "total_B": total / 1e9,
            "mla_per_layer": mla_params,
            "ffn_per_layer": dense_ffn_params,
        }


if __name__ == "__main__":
    config = NovaMind3BConfig()
    counts = config.count_parameters()
    print("=" * 60)
    print("NovaMind-3B Parameter Count")
    print("=" * 60)
    for k, v in counts.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.6f}")
        else:
            print(f"  {k:25s}: {v:>15,}")
    print("=" * 60)
