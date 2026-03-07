"""
Hybrid 3B Reasoning Model Configuration.

Architecture: GDN/MLA Hybrid + Dense SwiGLU FFN + RMSNorm + RoPE + MTP
  - 3:1 ratio of Gated DeltaNet (linear O(n)) to MLA (quadratic O(n²))
  - 20 GDN layers + 6 MLA layers = 26 total
  - ~3.7B total parameters (all activated per token — no MoE)

Target hardware: 2× NVIDIA L40S (2× 48 GB) via FSDP ZeRO-2
  - VRAM budget: ~22 GB/GPU with FSDP ZeRO-2 + activation checkpointing
    (weights 3.7GB + grads 7.4GB + Adam 14.8GB + activations ~14GB) / 2 GPUs
  - Fits comfortably in 48 GB per GPU
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NovaMind3BConfig:
    # --- Tokenizer ---
    vocab_size: int = 100352          # cl100k_base padded to multiple of 64

    # --- Transformer core ---
    hidden_dim: int = 3072            # 2× the 1B hidden (1536→3072)
    num_layers: int = 26              # 20 GDN + 6 MLA = 26 total
    max_seq_len: int = 8192
    dropout: float = 0.0

    # --- Hybrid Architecture ---
    # 3:1 ratio: every 4th layer uses MLA (full attention), rest use GDN
    use_hybrid: bool = True
    hybrid_attention_layers: List[int] = field(
        default_factory=lambda: [3, 7, 11, 15, 19, 23]   # 6 MLA layers
    )

    # --- Gated DeltaNet config (for non-MLA layers in hybrid) ---
    # Matches fla reference: key_dim = 0.75d, value_dim = 1.5d → 6d² params
    gdn_num_heads: int = 9            # H (keys/queries)
    gdn_head_dim: int = 256           # D_k per head (key_dim = 9×256 = 2304)
    gdn_expand_v: float = 2.0         # D_v = D_k × 2 = 512 (value_dim = 4608)
    gdn_use_gate: bool = True         # Output sigmoid gate + gated RMSNorm
    gdn_use_short_conv: bool = True   # Causal depthwise conv on Q, K, V
    gdn_conv_size: int = 4            # Conv kernel size

    # --- Multi-head Latent Attention (MLA) — for attention layers ---
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
    # Position interpolation scale factor for context extension.
    # Pretrain: 1.0 (no scaling, max_seq_len=8192).
    # SFT/DPO at 128K: set to 16.0 (= 131072 / 8192) so positions are
    # divided by 16 before RoPE, keeping them inside the pretrain distribution.
    rope_scale_factor: float = 1.0

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

        # GDN per layer (if hybrid)
        gdn_params = 0
        if self.use_hybrid:
            gdn_key_dim = self.gdn_num_heads * self.gdn_head_dim
            gdn_head_v = int(self.gdn_head_dim * self.gdn_expand_v)
            gdn_value_dim = self.gdn_num_heads * gdn_head_v
            gdn_params = (
                d * gdn_key_dim +           # q_proj
                d * gdn_key_dim +           # k_proj
                d * gdn_value_dim +         # v_proj
                d * self.gdn_num_heads +    # b_proj
                d * self.gdn_num_heads +    # a_proj
                self.gdn_num_heads +        # A_log
                self.gdn_num_heads +        # dt_bias
                gdn_value_dim * d           # o_proj
            )
            if self.gdn_use_gate:
                gdn_params += d * gdn_value_dim  # g_proj
                gdn_params += gdn_head_v         # o_norm_weight
            else:
                gdn_params += gdn_head_v         # o_norm
            if self.gdn_use_short_conv:
                gdn_params += (gdn_key_dim + gdn_key_dim + gdn_value_dim) * self.gdn_conv_size

        # Dense FFN (SwiGLU): gate + up + down
        dense_ffn_params = d * self.dense_intermediate * 3

        # Count layers by type
        if self.use_hybrid:
            n_mla = len(self.hybrid_attention_layers)
            n_gdn = self.num_layers - n_mla
        else:
            n_mla = self.num_layers
            n_gdn = 0

        mla_layer_total = mla_params + dense_ffn_params
        gdn_layer_total = gdn_params + dense_ffn_params
        total_layers = n_mla * mla_layer_total + n_gdn * gdn_layer_total

        # MoE layers (0 if fully dense)
        moe_ffn_params = 0
        total_moe = 0
        if self.n_routed_experts > 0 and self.num_moe_layers > 0:
            shared_expert_params = d * self.shared_expert_intermediate * 3 * self.n_shared_experts
            single_expert_params = d * self.expert_intermediate * 3
            router_params = d * self.n_routed_experts
            moe_ffn_params = shared_expert_params + single_expert_params * self.n_routed_experts + router_params
            total_moe = self.num_moe_layers * (moe_ffn_params - dense_ffn_params)

        # RMSNorm (2 per layer + 1 final)
        rmsnorm_params = (self.num_layers * 2 + 1) * d

        # Embedding
        embedding_params = self.vocab_size * d

        # MTP module (always uses MLA)
        mtp_params = 0
        if self.mtp_depth > 0:
            mtp_projection = 2 * d * d
            mtp_block = mla_params + dense_ffn_params
            mtp_rmsnorm = 3 * d
            mtp_params = (mtp_projection + mtp_block + mtp_rmsnorm) * self.mtp_depth

        total = embedding_params + total_layers + total_moe + rmsnorm_params + mtp_params

        return {
            "embedding": embedding_params,
            "mla_layers (×{})".format(n_mla): n_mla * mla_layer_total,
            "gdn_layers (×{})".format(n_gdn): n_gdn * gdn_layer_total,
            "moe_extra (0 if dense)": total_moe,
            "rmsnorm": rmsnorm_params,
            "mtp": mtp_params,
            "total": total,
            "total_B": total / 1e9,
            "mla_per_layer": mla_layer_total,
            "gdn_per_layer": gdn_layer_total if n_gdn > 0 else 0,
            "ffn_per_layer": dense_ffn_params,
        }


if __name__ == "__main__":
    config = NovaMind3BConfig()
    counts = config.count_parameters()
    print("=" * 60)
    print("NovaMind-3B Hybrid Parameter Count")
    print(f"  Architecture: {config.num_layers - len(config.hybrid_attention_layers)} GDN + "
          f"{len(config.hybrid_attention_layers)} MLA layers (3:1 ratio)")
    print("=" * 60)
    for k, v in counts.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.6f}")
        else:
            print(f"  {k:25s}: {v:>15,}")
    print("=" * 60)
