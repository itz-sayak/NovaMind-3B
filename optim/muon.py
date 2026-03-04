"""
Muon Optimizer - Momentum with Orthogonalization for Neural Networks.

Based on: https://github.com/KellerJordan/Muon

Muon is used for hidden layer weights (2D parameters).
AdamW is used for embeddings, output heads, biases, and normalization parameters.

The key idea: after computing the momentum, orthogonalize it using Newton-Schulz
iterations to get an approximately orthogonal update direction. This leads to
better optimization landscapes for training neural networks.
"""
import torch
from torch.optim import Optimizer
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization.
    
    Maps G to the nearest orthogonal matrix (in Frobenius norm sense).
    This is equivalent to computing U @ V^T from SVD G = U S V^T.
    
    Uses 5th-order Newton-Schulz coefficients for fast convergence.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.float()
    
    # Ensure the spectral norm is <= 1 for convergence
    X = X / (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if transposed:
        X = X.T
    
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon - Momentum + Orthogonalization optimizer.
    
    For hidden layer weights (2D tensors), applies Newton-Schulz orthogonalization
    to the momentum, giving updates that approximately preserve the spectral structure.
    
    For other parameters, falls back to standard AdamW.
    
    Args:
        muon_params: iterable of parameters to optimize with Muon
        lr: learning rate for Muon parameters (default: 0.02)
        momentum: momentum factor (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        adamw_params: iterable of parameters to optimize with AdamW
        adamw_lr: learning rate for AdamW parameters (default: 3e-4)
        adamw_betas: beta coefficients for AdamW (default: (0.9, 0.95))
        adamw_weight_decay: weight decay for AdamW (default: 0.01)
    """

    def __init__(
        self,
        muon_params=None,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_weight_decay=0.01,
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        
        params = []
        
        # Muon params (hidden layer weights, 2D)
        if muon_params is not None:
            muon_params = list(muon_params)
            params.append({
                "params": muon_params,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
            })
        
        # AdamW params (embeddings, biases, norms)
        if adamw_params is not None:
            adamw_params = list(adamw_params)
            params.append({
                "params": adamw_params,
                "use_muon": False,
                "lr": adamw_lr,
                "betas": adamw_betas,
                "weight_decay": adamw_weight_decay,
                "eps": adamw_eps,
            })
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        """Muon update for hidden layer 2D weights."""
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # Initialize state — keep momentum in fp32 for Newton-Schulz precision
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros(
                    grad.shape, dtype=torch.float32, device=grad.device
                )

            state["step"] += 1
            buf = state["momentum_buffer"]

            # Update momentum buffer (upcast grad to fp32)
            buf.mul_(momentum).add_(grad.float())

            if nesterov:
                update = grad + momentum * buf
            else:
                update = buf.clone()

            # Apply Newton-Schulz orthogonalization for 2D weights
            if p.ndim >= 2:
                # Reshape to 2D if needed (e.g., for conv layers)
                original_shape = update.shape
                if p.ndim > 2:
                    update = update.view(update.shape[0], -1)
                
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                if p.ndim > 2:
                    update = update.view(original_shape)
                
                # Scale by sqrt(max(m,n)/min(m,n)) for proper scaling
                # This gives built-in muP scaling
                h, w = p.shape[0], p.shape[1] if p.ndim >= 2 else 1
                scale = max(1, (h / w) ** 0.5)
                update.mul_(scale)

            # Weight decay (decoupled)
            if wd > 0:
                p.mul_(1 - lr * wd)

            # Apply update
            p.add_(update, alpha=-lr)

    def _adamw_step(self, group):
        """Standard AdamW update for non-hidden parameters."""
        lr = group["lr"]
        betas = group.get("betas", (0.9, 0.95))
        eps = group.get("eps", 1e-8)
        wd = group.get("weight_decay", 0.01)
        beta1, beta2 = betas

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                # fp32 states for numerical stability even with bf16 params
                state["exp_avg"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # Decoupled weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)

            # Adam update (in fp32, then cast back to param dtype)
            exp_avg.mul_(beta1).add_(grad.float(), alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

            p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)


def create_optimizer(model, config):
    """
    Create Muon optimizer with proper parameter grouping for NovaMind-3B.
    
    - Muon: All 2D hidden layer weights (attention projections, FFN weights, expert weights)
    - AdamW: Embeddings, output head, RMSNorm weights, router weights, biases
    """
    muon_params = []
    adamw_params = []
    
    # Categorize parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Always use AdamW for: embeddings, norms, biases, router gates
        if any(k in name for k in ["embedding", "final_norm", "attn_norm", "ffn_norm",
                                     "q_norm", "kv_norm", "h_norm", "emb_norm",
                                     ".weight" if "norm" in name else "___never___"]):
            if "norm" in name or "embedding" in name:
                adamw_params.append(param)
                continue
        
        if "router" in name or "gate" in name and "gate_proj" not in name:
            adamw_params.append(param)
            continue
        
        if param.ndim < 2:  # biases and 1D params
            adamw_params.append(param)
            continue
        
        # Everything else (2D hidden weights) uses Muon
        muon_params.append(param)
    
    optimizer = Muon(
        muon_params=muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        nesterov=config.muon_nesterov,
        ns_steps=config.muon_ns_steps,
        weight_decay=config.muon_weight_decay,
        adamw_params=adamw_params,
        adamw_lr=config.learning_rate,
        adamw_betas=(config.beta1, config.beta2),
        adamw_weight_decay=config.weight_decay,
    )
    
    muon_count = sum(p.numel() for p in muon_params)
    adamw_count = sum(p.numel() for p in adamw_params)
    print(f"Optimizer: Muon params: {muon_count:,} | AdamW params: {adamw_count:,}")
    
    return optimizer
