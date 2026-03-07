"""
Gated DeltaNet: Linear attention via the gated delta rule.

Based on "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(Yang, Kautz & Hatamizadeh, ICLR 2025).

Used in hybrid architectures at a 3:1 ratio with standard attention,
achieving large decoding throughput gains at long context lengths.

The gated delta rule combines two complementary mechanisms:
  - Gating  (Mamba2):  adaptive per-head state decay  α_t = exp(g_t)
  - Delta rule:  targeted memory write  β_t · k_t · (v_t − k_t^T · S_{t-1})

State update:  S_t = α_t · S_{t-1} + β_t · k_t · (v_t − k_t^T · S_{t-1})^T
Output:        o_t = q_t^T · S_t

Key advantages over standard softmax attention:
  - O(1) memory per token during inference (fixed-size state, no KV cache)
  - O(n) training with chunk-parallel algorithm (via fla Triton kernels)
  - Inherently causal (no attention mask needed)
  - Better length extrapolation and long-context efficiency

Uses optimized Triton kernels from fla (flash-linear-attention) when available.
Falls back to a correct but slower PyTorch recurrent computation otherwise.

Install fla for fast training:  pip install flash-linear-attention
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Optional: fla (flash-linear-attention) Triton kernels ──────────────────
# Provide chunk-parallel training (~5-10x faster than the PyTorch fallback).
# Install:  pip install flash-linear-attention
# Set FLA_DISABLE=1 to force the pure-PyTorch fallback (e.g. for smoke tests
# on hardware where Triton JIT compilation is unavailable or slow).
import os as _os
try:
    if _os.environ.get("FLA_DISABLE", "0") == "1":
        raise ImportError("FLA_DISABLE=1")
    # fla/__init__.py imports fla.layers and fla.models, which drag in
    # transformers → torchvision → bz2 → _bz2 (absent in some pyenv builds
    # compiled without libbz2 headers).  fla/ops/__init__.py imports every
    # ops submodule (abc, linear_attn, …) which we don't need.
    # Pre-stub both packages in sys.modules so their __init__.py files are
    # skipped entirely; subpackage lookups still work via __path__.
    import sys as _sys, types as _types, importlib.util as _ilu
    _fla_spec = _ilu.find_spec('fla')
    if _fla_spec is None:
        raise ImportError("flash-linear-attention not installed")
    import os.path as _osp
    _fla_root = _osp.dirname(_fla_spec.origin)
    for _pkg_name, _rel in (('fla', ''), ('fla.ops', 'ops')):
        if _pkg_name not in _sys.modules:
            _m = _types.ModuleType(_pkg_name)
            _m.__path__ = [_osp.join(_fla_root, _rel) if _rel else _fla_root]
            _m.__package__ = _pkg_name
            _sys.modules[_pkg_name] = _m
    del _fla_spec, _fla_root, _pkg_name, _rel, _m
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
    _FLA_AVAILABLE = True
except (ImportError, RuntimeError):
    # ImportError  → fla not installed, _bz2 missing, FLA_DISABLE=1, etc.
    # RuntimeError → triton.autotune fires on a CPU-only node (no CUDA driver)
    _FLA_AVAILABLE = False
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None


# ── FLA kernel mode ────────────────────────────────────────────────────────
# "chunk"     → chunk_gated_delta_rule (fastest, chunk-parallel forward+backward)
# "recurrent" → fused_recurrent_gated_delta_rule (uses different backward path)
# "off"       → pure-PyTorch recurrent fallback (correct but ~5-10× slower)
#
# Set by warmup_fla_kernels() at training startup.  Falls through chunk →
# recurrent → off, stopping at the first mode whose backward pass succeeds.
_fla_mode: str = "chunk" if _FLA_AVAILABLE else "off"


def warmup_fla_kernels(device: torch.device = None, verbose: bool = True):
    """Probe FLA Triton kernels with a tiny forward+backward and select the
    best working mode.  Must be called **before** the first training step so
    that a Triton compilation failure doesn't crash real training.

    Sets the module-level ``_fla_mode`` to 'chunk', 'recurrent', or 'off'.
    """
    global _fla_mode
    if not _FLA_AVAILABLE:
        _fla_mode = "off"
        return _fla_mode
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warn about conflicting FLA packages
    if verbose:
        try:
            import importlib.metadata as _im
            fla_dists = [d.metadata['Name'] for d in _im.distributions()
                         if 'fla' in d.metadata['Name'].lower()]
            if len(fla_dists) > 1:
                print(f"  WARNING: Multiple FLA packages installed: {fla_dists}")
                print("  This causes Triton kernel conflicts. Fix with:")
                print("    pip uninstall fla-core flash-linear-attention -y")
                print("    pip install git+https://github.com/fla-org/flash-linear-attention.git --force-reinstall --no-deps")
        except Exception:
            pass

    B, T, H, D_k, D_v = 1, 16, 2, 32, 32
    q = torch.randn(B, T, H, D_k, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(B, T, H, D_v, device=device, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(B, T, H, device=device, dtype=torch.bfloat16).abs().neg()  # < 0
    beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)

    for mode, kernel in (("chunk", chunk_gated_delta_rule),
                         ("recurrent", fused_recurrent_gated_delta_rule)):
        try:
            q_ = q.detach().clone().requires_grad_(True)
            k_ = k.detach().clone().requires_grad_(True)
            v_ = v.detach().clone().requires_grad_(True)
            o, _ = kernel(q=q_, k=k_, v=v_, g=g.to(q_.dtype), beta=beta,
                          output_final_state=False, use_qk_l2norm_in_kernel=True)
            o.sum().backward()
            _fla_mode = mode
            return _fla_mode
        except Exception as e:
            if verbose:
                print(f"  FLA probe '{mode}' failed: {type(e).__name__}: {e}")
            continue

    _fla_mode = "off"
    return _fla_mode


from model.attention import RMSNorm


# ────────────────────────────────────────────────────────────────────────────
# Short Convolution
# ────────────────────────────────────────────────────────────────────────────

class ShortConvolution(nn.Module):
    """Causal depthwise 1D convolution with SiLU activation.

    Provides short-range local context mixing before the delta rule operates.
    Critical for model quality — do not disable unless benchmarking.

    During inference (use_cache), maintains a sliding window of the last
    (kernel_size - 1) projected inputs for causal consistency.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,         # depthwise: each channel independent
            bias=False,
            padding=0,          # causal padding handled manually
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor = None,
        return_cache: bool = False,
    ) -> tuple:
        """
        Args:
            x:  (B, T, D) projected input (post-linear, pre-conv)
            cache:  (B, kernel_size-1, D) recent raw inputs, or None
            return_cache:  whether to output an updated cache

        Returns:
            y:  (B, T, D) convolved + SiLU activated output
            new_cache:  (B, kernel_size-1, D) or None
        """
        # Causal padding: prepend cached inputs (inference) or zeros (training)
        if cache is not None:
            x_padded = torch.cat([cache, x], dim=1)      # (B, K-1+T, D)
        else:
            x_padded = F.pad(x, (0, 0, self.kernel_size - 1, 0))  # zero-pad left

        # Depthwise conv1d: (B, D, K-1+T) → (B, D, T)
        y = self.conv(x_padded.transpose(1, 2)).transpose(1, 2)
        y = F.silu(y)

        new_cache = None
        if return_cache:
            # Last K-1 raw (pre-conv) inputs for the next forward call
            new_cache = x_padded[:, -(self.kernel_size - 1):, :].contiguous()

        return y, new_cache


# ────────────────────────────────────────────────────────────────────────────
# Gated DeltaNet Layer
# ────────────────────────────────────────────────────────────────────────────

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet attention layer (replaces standard attention in hybrid models).

    Parameter allocation with use_gate=True (standard):
        Q, K projections:  0.75 × d  each  →  1.5 d²
        V, G, O projections:  1.5 × d  each  →  4.5 d²
        Total ≈ 6 × hidden_dim²  per layer

    This matches the parameter budget of a standard transformer attention layer.

    Config requirements (all prefixed gdn_*):
        gdn_num_heads      – number of attention heads (H)
        gdn_head_dim       – per-head Q/K dimension (D_k)
        gdn_expand_v       – V expansion factor (head_v_dim = head_dim × expand_v)
        gdn_use_gate       – output sigmoid gate with RMSNorm
        gdn_use_short_conv – causal depthwise conv on Q, K, V
        gdn_conv_size      – conv kernel size (default 4)
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # ── Dimensions ────────────────────────────────────────────────────
        self.hidden_dim    = config.hidden_dim
        self.num_heads     = config.gdn_num_heads        # H
        self.head_dim      = config.gdn_head_dim         # D_k
        self.expand_v      = config.gdn_expand_v
        self.head_v_dim    = int(self.head_dim * self.expand_v)   # D_v
        self.key_dim       = self.num_heads * self.head_dim       # H × D_k
        self.value_dim     = self.num_heads * self.head_v_dim     # H × D_v
        self.use_gate      = config.gdn_use_gate
        self.use_short_conv = config.gdn_use_short_conv

        # ── Linear projections ────────────────────────────────────────────
        self.q_proj = nn.Linear(self.hidden_dim, self.key_dim,   bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.key_dim,   bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.value_dim, bias=False)

        # ── Delta-rule controls ───────────────────────────────────────────
        # β (beta): write strength per head, sigmoid → [0, 1]
        self.b_proj = nn.Linear(self.hidden_dim, self.num_heads, bias=False)

        # g (gate): state decay, computed as  -A · softplus(a(x) + dt_bias)
        self.a_proj = nn.Linear(self.hidden_dim, self.num_heads, bias=False)

        # Learnable decay magnitude (Mamba2-style A parameter)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt_bias: Mamba2-style initialization of gate bias
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── Short convolutions (depthwise causal, crucial for quality) ────
        if self.use_short_conv:
            ks = config.gdn_conv_size
            self.q_conv = ShortConvolution(self.key_dim,   ks)
            self.k_conv = ShortConvolution(self.key_dim,   ks)
            self.v_conv = ShortConvolution(self.value_dim, ks)
        else:
            warnings.warn(
                "ShortConvolution is critical for Gated DeltaNet quality. "
                "Do not disable (gdn_use_short_conv=False) unless benchmarking."
            )

        # ── Output gate + normalization ───────────────────────────────────
        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_dim, self.value_dim, bias=False)
            # Per-head RMSNorm weight (combined with sigmoid gating)
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
            self.norm_eps = config.rms_norm_eps
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)

        # ── Output projection ─────────────────────────────────────────────
        self.o_proj = nn.Linear(self.value_dim, self.hidden_dim, bias=False)

        self._warned_fallback = False

    # ── Helpers ────────────────────────────────────────────────────────────

    def _gated_rmsnorm(self, o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """RMSNorm(o) ⊙ sigmoid(g), applied per-head dimension."""
        rms = torch.rsqrt(o.float().pow(2).mean(-1, keepdim=True) + self.norm_eps)
        o_normed = (o.float() * rms).to(o.dtype) * self.o_norm_weight
        return o_normed * g.sigmoid()

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        """
        Args:
            x:  (B, T, D) hidden states
            attention_mask:  ignored (GDN is inherently causal)
            past_kv:  dict with 'recurrent_state', 'conv_state_{q,k,v}'
            use_cache:  return updated state for autoregressive generation

        Returns:
            (output, new_cache)  where output is (B, T, D)
        """
        B, T, D = x.shape

        # ── Extract cached states ──────────────────────────────────────────
        initial_state = None
        conv_q = conv_k = conv_v = None
        if past_kv is not None:
            initial_state = past_kv.get('recurrent_state')
            conv_q = past_kv.get('conv_state_q')
            conv_k = past_kv.get('conv_state_k')
            conv_v = past_kv.get('conv_state_v')

        # ── Project + convolve Q, K, V ─────────────────────────────────────
        if self.use_short_conv:
            q, new_conv_q = self.q_conv(
                self.q_proj(x), cache=conv_q, return_cache=use_cache)
            k, new_conv_k = self.k_conv(
                self.k_proj(x), cache=conv_k, return_cache=use_cache)
            v, new_conv_v = self.v_conv(
                self.v_proj(x), cache=conv_v, return_cache=use_cache)
        else:
            q = F.silu(self.q_proj(x))
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))
            new_conv_q = new_conv_k = new_conv_v = None

        # ── Reshape to multi-head ──────────────────────────────────────────
        q = q.view(B, T, self.num_heads, self.head_dim)      # (B, T, H, D_k)
        k = k.view(B, T, self.num_heads, self.head_dim)      # (B, T, H, D_k)
        v = v.view(B, T, self.num_heads, self.head_v_dim)    # (B, T, H, D_v)

        # ── Compute β (write strength) and g (decay gate) ─────────────────
        beta = self.b_proj(x).sigmoid()                       # (B, T, H) ∈ [0,1]
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias
        )                                                      # (B, T, H) < 0

        # ── Gated delta-rule kernel ────────────────────────────────────────
        if _fla_mode == "chunk":
            o, final_state = chunk_gated_delta_rule(
                q=q, k=k, v=v,
                g=g.to(q.dtype),
                beta=beta,
                initial_state=initial_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        elif _fla_mode == "recurrent":
            o, final_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v,
                g=g.to(q.dtype),
                beta=beta,
                initial_state=initial_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            # Slow path: pure-PyTorch recurrent (correct but ~5-10x slower)
            if not self._warned_fallback and self.training:
                warnings.warn(
                    "fla (flash-linear-attention) not installed — using slow PyTorch "
                    "recurrent fallback for GatedDeltaNet. Training will be significantly "
                    "slower. Install:  pip install flash-linear-attention",
                    stacklevel=2,
                )
                self._warned_fallback = True

            # L2-normalize Q, K (fla does this internally via use_qk_l2norm_in_kernel)
            q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

            o, final_state = self._recurrent_forward(
                q, k, v, g, beta, initial_state, use_cache
            )

        # ── Output gate + projection ──────────────────────────────────────
        if self.use_gate:
            g_out = self.g_proj(x).view(B, T, self.num_heads, self.head_v_dim)
            o = self._gated_rmsnorm(o, g_out)
        else:
            o = self.o_norm(o)

        o = o.reshape(B, T, self.value_dim)
        output = self.o_proj(o)

        # ── Build cache ────────────────────────────────────────────────────
        new_cache = None
        if use_cache:
            new_cache = {
                'recurrent_state': final_state,
                'conv_state_q': new_conv_q,
                'conv_state_k': new_conv_k,
                'conv_state_v': new_conv_v,
            }

        return output, new_cache

    # ── Pure-PyTorch fallback ──────────────────────────────────────────────

    def _recurrent_forward(self, q, k, v, g, beta, initial_state, output_final_state):
        """
        Recurrent gated delta rule in pure PyTorch.

        Per time-step:
            retrieved = k_t^T @ S              (current stored value for key k_t)
            error     = v_t − retrieved        (delta between desired and stored)
            S_t       = α_t · S_{t-1} + β_t · k_t ⊗ error
            o_t       = q_t^T @ S_t

        Where α_t = exp(g_t), g_t < 0  →  α_t ∈ (0, 1].
        """
        B, T, H, D_k = q.shape
        D_v = v.shape[-1]
        device = q.device
        orig_dtype = q.dtype

        # Work in float32 for numerical stability
        q, k, v = q.float(), k.float(), v.float()
        g, beta = g.float(), beta.float()

        # State: (B, H, D_k, D_v) — the associative memory matrix per head
        if initial_state is not None:
            S = initial_state.float()
        else:
            S = torch.zeros(B, H, D_k, D_v, device=device)

        outputs = []
        for t in range(T):
            q_t = q[:, t]        # (B, H, D_k)
            k_t = k[:, t]        # (B, H, D_k)
            v_t = v[:, t]        # (B, H, D_v)
            g_t = g[:, t]        # (B, H)
            beta_t = beta[:, t]  # (B, H)

            # Decay factor: α = exp(g), g < 0 → α ∈ (0, 1]
            alpha = g_t.exp()                            # (B, H)

            # Delta rule: read → error → write
            retrieved = torch.einsum('bhk,bhkv->bhv', k_t, S)  # (B, H, D_v)
            error = v_t - retrieved                              # (B, H, D_v)

            # State update:  S = α · S + β · k ⊗ error
            S = (alpha[:, :, None, None] * S
                 + beta_t[:, :, None, None]
                 * k_t[:, :, :, None] * error[:, :, None, :])

            # Read output:  o = q^T @ S
            o_t = torch.einsum('bhk,bhkv->bhv', q_t, S)        # (B, H, D_v)
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)  # (B, T, H, D_v)
        final_state = S if output_final_state else None
        return o.to(orig_dtype), final_state
