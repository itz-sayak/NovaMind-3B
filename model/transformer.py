"""
Full NovaMind inspired Transformer model at 1B scale.

Components:
  - Token Embedding (shared with output head)
  - N Transformer Layers (MLA + FFN/MoE)
  - Multi-Token Prediction module
  - RMSNorm throughout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.attention import MultiHeadLatentAttention, RMSNorm
from model.moe import NovaMindMoELayer, DenseFFN


@torch.compiler.disable
def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 512,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Compute linear-projection + cross-entropy in chunks.

    Instead of materialising the full (N, V) logits tensor in float32
    (~3 GiB for seq 8192 and vocab 100k), we process *chunk_size* tokens
    at a time and accumulate the loss.

    Args:
        hidden: (N, D)  hidden states  (bf16 ok)
        weight: (V, D)  output head weight
        targets: (N,)   target token ids
        chunk_size: tokens per chunk
        ignore_index: label to ignore
    """
    N = hidden.shape[0]
    # Accumulate in float32 — bf16 overflows at ~65504, but
    # sum over 8192 tokens * CE~11.5 ≈ 94k which exceeds bf16 max.
    total_loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
    total_valid = 0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tgt = targets[start:end]
        valid = (tgt != ignore_index).sum().item()
        if valid == 0:
            continue
        logits = F.linear(hidden[start:end], weight)  # (chunk, V)
        chunk_loss = F.cross_entropy(
            logits, tgt, ignore_index=ignore_index, reduction="sum"
        )
        total_loss = total_loss + chunk_loss
        total_valid += valid
        del logits, chunk_loss  # free immediately

    return (total_loss / max(total_valid, 1)).to(hidden.dtype)


class TransformerBlock(nn.Module):
    """Single transformer block: RMSNorm -> MLA -> residual -> RMSNorm -> FFN/MoE -> residual."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        # Multi-head Latent Attention
        self.attention = MultiHeadLatentAttention(config)

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        # FFN: Dense for first N layers, MoE for the rest
        if layer_idx < config.num_dense_layers:
            self.ffn = DenseFFN(config)
        else:
            self.ffn = NovaMindMoELayer(config, layer_idx)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        """
        Returns: (output, balance_loss, expert_counts, new_kv_cache)
        """
        # Attention block with residual
        normed = self.attn_norm(x)
        attn_out, new_cache = self.attention(
            normed, attention_mask=attention_mask,
            past_kv=past_kv, use_cache=use_cache
        )
        x = x + self.dropout(attn_out)

        # FFN block with residual
        normed = self.ffn_norm(x)
        ffn_out, balance_loss, expert_counts = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, balance_loss, expert_counts, new_cache


class MTPModule(nn.Module):
    """
    Multi-Token Prediction module (1 depth).

    Given the last layer's hidden states and the next token embeddings,
    predicts an additional future token at each position.
    """

    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim

        # Projection: [RMSNorm(h); RMSNorm(Emb(next_token))] -> h'
        self.h_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.emb_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.projection = nn.Linear(2 * d, d, bias=False)

        # Lightweight transformer block for MTP
        self.block = TransformerBlock(config, layer_idx=0)  # Use dense FFN

        self.final_norm = RMSNorm(d, eps=config.rms_norm_eps)

    def forward(self, hidden_states, next_token_embeddings, output_head):
        """
        Args:
            hidden_states: (B, T, D) from the main model
            next_token_embeddings: (B, T, D) embeddings of tokens at positions [1, 2, ..., T]
            output_head: the shared output head (embedding weight)

        Returns:
            mtp_logits: (B, T, vocab_size) predictions for positions [2, 3, ..., T+1]
            mtp_loss_extra: balance loss from the MTP block
        """
        # Combine hidden state with next token embedding
        h_normed = self.h_norm(hidden_states)
        emb_normed = self.emb_norm(next_token_embeddings)
        combined = torch.cat([h_normed, emb_normed], dim=-1)  # (B, T, 2D)
        h_prime = self.projection(combined)  # (B, T, D)

        # Pass through transformer block
        h_mtp, balance_loss, _, _ = self.block(h_prime)

        # Project to vocab using shared output head
        h_mtp = self.final_norm(h_mtp)
        # During training, avoid materializing (B, T, V) logit tensor (see MTP caller)
        return h_mtp, balance_loss


class NovaMind3B(nn.Module):
    """
    Dense 1B Reasoning Model.

    Architecture:
    - Token Embedding (tied with output head)
    - 24 Transformer layers: MLA + Dense SwiGLU FFN (d_ff=6144)
    - RMSNorm final layer
    - Multi-Token Prediction module (MTP depth=1, λ=0.3)

    ~1.09B total parameters, all activated per token.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Output head (tied with embedding)
        if not config.tie_word_embeddings:
            self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Multi-Token Prediction
        self.mtp = None
        if config.mtp_depth > 0:
            self.mtp = MTPModule(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to output projections (like GPT-2)
        for layer in self.layers:
            if hasattr(layer.attention, 'W_O'):
                nn.init.normal_(layer.attention.W_O.weight,
                              std=config.init_std / (2 * config.num_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def get_output_head_weight(self):
        """Get the output head weight (tied or separate)."""
        if self.config.tie_word_embeddings:
            return self.embedding.weight
        return self.output_head.weight

    def forward(self, input_ids, targets=None, use_cache=False, past_kv_list=None):
        """
        Args:
            input_ids: (B, T) token indices
            targets: (B, T) target token indices for loss computation
            use_cache: for autoregressive generation
            past_kv_list: list of past KV caches per layer

        Returns:
            dict with keys: logits, loss, balance_loss, mtp_loss
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embedding(input_ids)  # (B, T, D)

        # Process through transformer layers
        total_balance_loss = torch.tensor(0.0, device=device)
        all_expert_counts = []
        new_kv_list = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None

            if self.config.gradient_checkpointing and self.training and not use_cache:
                # Gradient checkpointing for memory savings
                x, balance_loss, expert_counts, new_cache = checkpoint(
                    layer, x, None, None, False,
                    use_reentrant=False
                )
            else:
                x, balance_loss, expert_counts, new_cache = layer(
                    x, past_kv=past_kv, use_cache=use_cache
                )

            total_balance_loss = total_balance_loss + balance_loss
            if expert_counts is not None:
                all_expert_counts.append((i, expert_counts))
            if use_cache:
                new_kv_list.append(new_cache)

        # Final norm
        x = self.final_norm(x)

        # Compute logits
        output_weight = self.get_output_head_weight()

        result = {
            "logits": None,
            "loss": None,
            "balance_loss": total_balance_loss,
            "mtp_loss": torch.tensor(0.0, device=device),
            "expert_counts": all_expert_counts,
        }

        if use_cache:
            # Inference: need full logits
            logits = F.linear(x, output_weight)  # (B, T, vocab_size)
            result["logits"] = logits
            result["past_kv_list"] = new_kv_list

        # Compute losses if targets provided
        if targets is not None:
            # Main next-token prediction loss — chunked to avoid 3 GiB float32 peak.
            # targets is already shifted (dataset returns y = chunk[1:]), so position i
            # predicts targets[i] directly — no additional shift needed here.
            shift_hidden  = x.contiguous().view(-1, x.size(-1))  # (B*T, D)
            shift_targets = targets.contiguous().view(-1)         # (B*T,)
            loss = chunked_cross_entropy(shift_hidden, output_weight, shift_targets)
            result["loss"] = loss

            # Also expose logits if not already set (needed in rare eval-with-targets paths)
            if result["logits"] is None and not self.training:
                logits = F.linear(x, output_weight)
                result["logits"] = logits

            # Multi-Token Prediction loss
            if self.mtp is not None and self.training:
                if T > 2:
                    mtp_hidden = x[:, :-2, :]                         # (B, T-2, D) positions 0..T-3
                    mtp_next_emb = self.embedding(input_ids[:, 1:-1])  # (B, T-2, D) next-token embeds
                    mtp_targets = targets[:, 1:-1]                     # (B, T-2) = chunk[2:T], 2 steps ahead

                    mtp_h, mtp_balance_loss = self.mtp(
                        mtp_hidden, mtp_next_emb, output_weight
                    )

                    # Chunked cross-entropy for MTP logits
                    mtp_h_flat = mtp_h.contiguous().view(-1, mtp_h.size(-1))
                    mtp_tgt_flat = mtp_targets.contiguous().view(-1)
                    mtp_loss = chunked_cross_entropy(
                        mtp_h_flat, output_weight, mtp_tgt_flat
                    )
                    result["mtp_loss"] = mtp_loss
                    total_balance_loss = total_balance_loss + mtp_balance_loss

            result["balance_loss"] = total_balance_loss

        return result

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9):
        """Autoregressive generation with KV cache."""
        self.eval()
        B = input_ids.shape[0]

        past_kv_list = None
        for _ in range(max_new_tokens):
            # Forward pass with cache
            if past_kv_list is None:
                idx_cond = input_ids
            else:
                idx_cond = input_ids[:, -1:]

            result = self(idx_cond, use_cache=True, past_kv_list=past_kv_list)
            past_kv_list = result["past_kv_list"]
            logits = result["logits"][:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
