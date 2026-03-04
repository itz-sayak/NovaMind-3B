"""
NovaMindMoE: Mixture-of-Experts with shared experts and auxiliary-loss-free
load balancing, inspired by NovaMind.

Key features:
  - Fine-grained experts (smaller but more numerous)
  - 1 shared expert always activated + K routed experts selected per token
  - Sigmoid gating with top-K normalization
  - Auxiliary-loss-free load balancing via dynamic bias terms
  - Complementary sequence-wise balance loss (very small alpha)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(gate) * up, then down-project."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """A single FFN expert using SwiGLU."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_dim, intermediate_dim)

    def forward(self, x):
        return self.ffn(x)


class MoERouter(nn.Module):
    """
    Router for NovaMindMoE with auxiliary-loss-free load balancing.

    Uses sigmoid gating (not softmax) with top-K selection and normalization.
    Maintains per-expert bias terms for load balancing that don't affect gradients.
    """

    def __init__(self, hidden_dim: int, n_experts: int, n_activated: int,
                 aux_loss_free: bool = True, bias_update_speed: float = 0.001,
                 balance_loss_alpha: float = 0.0001):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed
        self.balance_loss_alpha = balance_loss_alpha

        # Expert centroids for computing affinity
        self.gate = nn.Linear(hidden_dim, n_experts, bias=False)

        # Auxiliary-loss-free: per-expert bias for routing (not learned by gradient)
        if aux_loss_free:
            self.register_buffer(
                "expert_bias", torch.zeros(n_experts), persistent=True
            )

    def forward(self, x):
        """
        Args:
            x: (B*T, D) flattened token representations

        Returns:
            gate_values: (B*T, n_activated) normalized gating weights
            expert_indices: (B*T, n_activated) selected expert indices
            balance_loss: scalar auxiliary balance loss
        """
        # Compute affinity scores using sigmoid (not softmax, following NovaMind)
        affinity = torch.sigmoid(self.gate(x))  # (B*T, n_experts)

        # For routing decisions, add bias (doesn't affect gating values)
        if self.aux_loss_free:
            routing_scores = affinity + self.expert_bias.unsqueeze(0)
        else:
            routing_scores = affinity

        # Top-K selection
        topk_values, topk_indices = torch.topk(
            routing_scores, self.n_activated, dim=-1
        )  # both (B*T, n_activated)

        # Gating values from ORIGINAL affinity (not biased), then normalize
        gate_values = torch.gather(affinity, 1, topk_indices)  # (B*T, n_activated)
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute complementary sequence-wise balance loss
        balance_loss = self._compute_balance_loss(affinity, topk_indices)

        return gate_values, topk_indices, balance_loss

    def _compute_balance_loss(self, affinity, topk_indices):
        """Complementary sequence-wise balance loss (Eq. 17-20 in paper)."""
        if self.balance_loss_alpha == 0:
            return torch.tensor(0.0, device=affinity.device)

        T = affinity.shape[0]

        # f_i: fraction of tokens routed to expert i (scaled)
        # Create one-hot and sum over tokens
        expert_mask = F.one_hot(topk_indices, self.n_experts).float()  # (T, K, E)
        expert_mask = expert_mask.sum(dim=1)  # (T, E)
        f = expert_mask.sum(dim=0) * self.n_experts / (self.n_activated * T)  # (E,)

        # P_i: mean affinity for expert i, normalized
        affinity_norm = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-9)
        P = affinity_norm.mean(dim=0)  # (E,)

        balance_loss = self.balance_loss_alpha * (f * P).sum()
        return balance_loss

    @torch.no_grad()
    def update_expert_bias(self, expert_counts):
        """
        Update expert bias based on load statistics (called at end of each step).
        If an expert is overloaded, decrease its bias; if underloaded, increase.

        Args:
            expert_counts: (n_experts,) number of tokens routed to each expert
        """
        if not self.aux_loss_free:
            return

        avg_count = expert_counts.float().mean()
        overloaded = expert_counts > avg_count
        underloaded = expert_counts < avg_count

        self.expert_bias[overloaded] -= self.bias_update_speed
        self.expert_bias[underloaded] += self.bias_update_speed


class NovaMindMoELayer(nn.Module):
    """
    NovaMindMoE Feed-Forward layer with shared + routed experts.

    Output: u_t + sum(shared_expert(u_t)) + sum(g_i * routed_expert_i(u_t))
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            Expert(config.hidden_dim, config.shared_expert_intermediate)
            for _ in range(config.n_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            Expert(config.hidden_dim, config.expert_intermediate)
            for _ in range(config.n_routed_experts)
        ])

        # Router
        self.router = MoERouter(
            hidden_dim=config.hidden_dim,
            n_experts=config.n_routed_experts,
            n_activated=config.n_activated_experts,
            aux_loss_free=config.aux_loss_free,
            bias_update_speed=config.bias_update_speed,
            balance_loss_alpha=config.balance_loss_alpha,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) input

        Returns:
            output: (B, T, D)
            balance_loss: scalar
            expert_counts: (n_routed_experts,) for bias update
        """
        B, T, D = x.shape
        residual = x

        # Shared experts (always applied)
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # Flatten for routing
        x_flat = x.view(B * T, D)

        # Route tokens to experts
        gate_values, expert_indices, balance_loss = self.router(x_flat)

        # Compute routed expert outputs using scatter/gather
        routed_output = torch.zeros_like(x_flat)

        # Track expert load for bias update
        expert_counts = torch.zeros(
            self.config.n_routed_experts, device=x.device, dtype=torch.long
        )

        # Process each activated expert slot
        for k in range(self.config.n_activated_experts):
            expert_idx = expert_indices[:, k]  # (B*T,)
            gate_k = gate_values[:, k].unsqueeze(-1)  # (B*T, 1)

            # Count tokens per expert
            for e in range(self.config.n_routed_experts):
                mask = (expert_idx == e)
                expert_counts[e] += mask.sum()

                if mask.any():
                    tokens = x_flat[mask]  # (num_tokens, D)
                    expert_output = self.routed_experts[e](tokens)
                    routed_output[mask] += gate_k[mask] * expert_output

        routed_output = routed_output.view(B, T, D)

        # Combine: shared + routed (NO residual — TransformerBlock handles that)
        output = shared_output + routed_output

        return output, balance_loss, expert_counts


class DenseFFN(nn.Module):
    """Dense Feed-Forward Network using SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.ffn = SwiGLU(config.hidden_dim, config.dense_intermediate)

    def forward(self, x):
        # Returns ffn output WITHOUT residual — TransformerBlock adds residual.
        return self.ffn(x), torch.tensor(0.0, device=x.device), None
