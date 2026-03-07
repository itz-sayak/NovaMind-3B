"""
Multi-head Latent Attention (MLA) from NovaMind.

Key idea: low-rank joint compression of K and V into a compressed latent vector
to drastically reduce KV cache while maintaining MHA-level performance.

Queries are also compressed for reduced activation memory during training.
RoPE is applied to decoupled key/query components.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FlashAttention-2 — optional but strongly recommended for training throughput.
# Requires: pip install flash-attn  (pre-built wheel for torch2.5+cu12)
# Falls back transparently to torch.nn.functional.scaled_dot_product_attention.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 rope_scale_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_scale_factor = rope_scale_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        # Divide positions by rope_scale_factor to implement position
        # interpolation for context extension (Chen et al. 2023).
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype) / self.rope_scale_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply RoPE to input tensor."""
    # x: (B, n_heads, T, d_rope)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_rope)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) as described in NovaMind.

    Core idea:
    - Compress K,V jointly into a low-rank latent vector c_KV
    - Compress Q into a low-rank latent vector c_Q
    - Use decoupled RoPE on separate key/query projections
    - Only cache c_KV and k_R during generation (much smaller than full KV)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_kv_comp = config.d_kv_comp
        self.d_q_comp = config.d_q_comp
        self.d_rope = config.d_rope
        self.hidden_dim = config.hidden_dim
        self.head_dim_total = self.n_heads * self.d_head

        # --- KV compression ---
        # h_t -> c_KV (down-projection)
        self.W_DKV = nn.Linear(self.hidden_dim, self.d_kv_comp, bias=False)
        # c_KV -> K (up-projection, produces all heads)
        self.W_UK = nn.Linear(self.d_kv_comp, self.head_dim_total, bias=False)
        # c_KV -> V (up-projection, produces all heads)
        self.W_UV = nn.Linear(self.d_kv_comp, self.head_dim_total, bias=False)

        # --- Decoupled RoPE key ---
        # h_t -> k_R (for RoPE, shared across heads but broadcast)
        self.W_KR = nn.Linear(self.hidden_dim, self.d_rope, bias=False)

        # --- Query compression ---
        # h_t -> c_Q (down-projection)
        self.W_DQ = nn.Linear(self.hidden_dim, self.d_q_comp, bias=False)
        # c_Q -> Q (up-projection)
        self.W_UQ = nn.Linear(self.d_q_comp, self.head_dim_total, bias=False)
        # c_Q -> q_R (decoupled RoPE queries, per head)
        self.W_QR = nn.Linear(self.d_q_comp, self.d_rope * self.n_heads, bias=False)

        # --- Output projection ---
        self.W_O = nn.Linear(self.head_dim_total, self.hidden_dim, bias=False)

        # --- RoPE ---
        self.rotary_emb = RotaryEmbedding(
            self.d_rope,
            max_seq_len=config.max_seq_len,
            base=getattr(config, "rope_base", 500_000.0),
            rope_scale_factor=getattr(config, "rope_scale_factor", 1.0),
        )

        # Scaling factors (following NovaMind: additional RMSNorm after compressed latents)
        self.q_norm = RMSNorm(self.d_q_comp, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.d_kv_comp, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.d_head + self.d_rope)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        """
        Args:
            x: (B, T, D) input hidden states
            attention_mask: optional causal mask
            past_kv: cached (c_KV, k_R) for autoregressive generation
            use_cache: whether to return updated cache
        """
        B, T, D = x.shape

        # --- Query path ---
        c_Q = self.q_norm(self.W_DQ(x))   # (B, T, d_q_comp)
        q_C = self.W_UQ(c_Q)               # (B, T, n_h * d_h)
        q_R = self.W_QR(c_Q)               # (B, T, n_h * d_rope)

        # Reshape for multi-head
        q_C = q_C.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_h, T, d_h)
        q_R = q_R.view(B, T, self.n_heads, self.d_rope).transpose(1, 2)  # (B, n_h, T, d_rope)

        # --- KV path ---
        c_KV = self.kv_norm(self.W_DKV(x))  # (B, T, d_kv_comp)
        k_C = self.W_UK(c_KV)                # (B, T, n_h * d_h)
        v_C = self.W_UV(c_KV)                # (B, T, n_h * d_h)

        # Decoupled RoPE key (shared across heads, then broadcast)
        k_R = self.W_KR(x)                   # (B, T, d_rope)

        # Handle KV cache for generation
        if past_kv is not None:
            past_c_KV, past_k_R = past_kv
            c_KV = torch.cat([past_c_KV, c_KV], dim=1)
            k_R = torch.cat([past_k_R, k_R], dim=1)
            # Recompute k_C, v_C from full c_KV
            k_C = self.W_UK(c_KV)
            v_C = self.W_UV(c_KV)

        new_cache = (c_KV, k_R) if use_cache else None

        S = k_C.shape[1]  # total sequence length including cache

        # Reshape K, V
        k_C = k_C.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_h, S, d_h)
        v_C = v_C.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_h, S, d_h)

        # Broadcast k_R to all heads
        k_R = k_R.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, n_h, S, d_rope)

        # --- Apply RoPE to decoupled components ---
        cos, sin = self.rotary_emb(S)
        # For queries, apply RoPE to positions [S-T : S]
        q_cos, q_sin = cos[S - T:S], sin[S - T:S]
        q_R = apply_rotary_pos_emb(q_R, q_cos, q_sin)
        k_R = apply_rotary_pos_emb(k_R, cos[:S], sin[:S])

        # --- Concatenate content and RoPE components ---
        # q = [q_C; q_R], k = [k_C; k_R]
        q = torch.cat([q_C, q_R], dim=-1)   # (B, n_h, T, d_h + d_rope)
        k = torch.cat([k_C, k_R], dim=-1)   # (B, n_h, S, d_h + d_rope)

        # ── Attention kernel selection ─────────────────────────────────────
        # FlashAttention-2: O(T) memory, fused CUDA kernel, ~3× faster on L40S.
        # Expects layout (B, T, n_heads, d_head); requires identical head dim for q/k/v.
        # MLA uses d_head+d_rope=192 for QK but d_head=128 for V → zero-pad V to 192,
        # run flash_attn, then slice back. Zero-padding is exact: attn_w @ [v|0] = [attn_w@v|0].
        # Falls back to torch SDPA (still fused, just slower) when:
        #   - flash-attn not installed, or
        #   - an explicit additive attention_mask is provided (inference w/ mask).
        dropout_p = self.attn_dropout.p if self.training else 0.0

        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and attention_mask is None          # flash handles causal natively
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash:
            # Transpose from heads-first (B, n_h, T, d) → sequence-first (B, T, n_h, d)
            # .contiguous() is required by the CUDA kernel.
            q_fa = q.transpose(1, 2).contiguous()    # (B, T,  n_h, d_h+d_rope=192)
            k_fa = k.transpose(1, 2).contiguous()    # (B, S,  n_h, d_h+d_rope=192)
            v_fa = v_C.transpose(1, 2).contiguous()  # (B, S,  n_h, d_h=128)
            # flash_attn_func requires q/k/v to share the same head dim.
            # Pad V from d_h (128) → d_h+d_rope (192) with zeros; slice back after.
            # Zero-pad is safe: attn_weights @ [v | 0] = [attn_weights @ v | 0].
            pad = q_fa.shape[-1] - v_fa.shape[-1]   # 192 - 128 = 64
            if pad > 0:
                v_fa = F.pad(v_fa, (0, pad))         # (B, S,  n_h, 192)
            # flash_attn_func returns (B, T, n_h, 192)
            attn_output = _flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=dropout_p,
                softmax_scale=self.scale,
                causal=True,
            )
            # Discard the zero-padded tail → (B, T, n_h, d_h=128)
            attn_output = attn_output[..., :self.d_head].reshape(B, T, self.head_dim_total)
        else:
            # Fallback: PyTorch 2.x SDPA (dispatches to memory-efficient / math kernel).
            # Still uses O(T) memory via chunked attention; no explicit mask matrix built.
            if attention_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v_C,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                    scale=self.scale,
                )
            else:
                # Additive mask (0 or -inf) provided — used during KV-cache inference.
                attn_output = F.scaled_dot_product_attention(
                    q, k, v_C,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=self.scale,
                )
            # attn_output: (B, n_h, T, d_h) → (B, T, n_h*d_h)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.head_dim_total)
        output = self.W_O(attn_output)  # (B, T, D)

        return output, new_cache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight
