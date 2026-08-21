"""
Sinkhorn-Knopp Algorithm for Optimal Transport.

Implements:
1. Standard entropic OT (Eq. 8 in paper)
2. Extended OT with molecular similarity constraints (Eq. 7, Algorithm 1)

Reference: Section 2.3, Algorithm 1 of the paper.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def sinkhorn_knopp(
    C: torch.Tensor,              # (M, N) cost matrix
    r: Optional[torch.Tensor] = None,  # (M,) source marginal
    c: Optional[torch.Tensor] = None,  # (N,) target marginal
    eps: float = 0.01,            # entropic regularization
    max_iter: int = 100,          # Sinkhorn iterations
    tol: float = 1e-6,           # convergence tolerance
) -> torch.Tensor:
    """
    Solve entropic OT: min_T <C, T> + eps * H(T)
    s.t. T @ 1 = r, T.T @ 1 = c

    Returns: T (M, N) optimal transport plan
    """
    M, N = C.shape
    device = C.device

    if r is None:
        r = torch.ones(M, device=device) / M
    if c is None:
        c = torch.ones(N, device=device) / N

    # Gibbs kernel
    K = torch.exp(-C / eps)  # (M, N)

    u = torch.ones(M, device=device)
    v = torch.ones(N, device=device)

    for _ in range(max_iter):
        u_prev = u.clone()
        u = r / (K @ v)
        v = c / (K.T @ u)

        # Check convergence
        if torch.norm(u - u_prev) / (torch.norm(u) + 1e-8) < tol:
            break

    T = torch.diag(u) @ K @ torch.diag(v)
    return T


def sinkhorn_with_similarity(
    S: torch.Tensor,              # (M, N) score matrix
    Sim: torch.Tensor,            # (M, M) molecular similarity matrix
    eps: float = 0.01,            # entropic regularization
    lam: float = 0.1,            # similarity weight λ
    eta: float = 1.0,            # gradient step size η
    max_iter: int = 50,          # outer iterations
    sinkhorn_iter: int = 50,     # inner Sinkhorn iterations
) -> torch.Tensor:
    """
    OT with similarity constraints (Algorithm 1).

    Alternates between:
    1. Sinkhorn-Knopp iteration to enforce marginal constraints
    2. Gradient update to enforce molecular similarity consistency

    Args:
        S: (M, N) predicted interaction scores (higher = more likely)
        Sim: (M, M) molecular similarity matrix (e.g., Tanimoto fingerprint similarity)
        eps: Sinkhorn entropic regularization
        lam: weight for similarity constraint
        eta: learning rate for similarity gradient step
        max_iter: number of outer iterations

    Returns:
        T: (M, N) optimal transport plan (pseudo-label matrix)
    """
    M, N = S.shape
    device = S.device

    # Cost matrix: C_ij = 1 - S_ij
    C = 1.0 - S
    C = C.clamp(min=0)

    # Uniform marginals
    r = torch.ones(M, device=device) / M
    c = torch.ones(N, device=device) / N

    # Initialize
    K = torch.exp(-C / eps)
    u = torch.ones(M, device=device)
    v = torch.ones(N, device=device)

    for t in range(max_iter):
        # Step 1: Sinkhorn-Knopp iterations
        for _ in range(sinkhorn_iter):
            u = r / (K @ v + 1e-10)
            v = c / (K.T @ u + 1e-10)

        T = torch.diag(u) @ K @ torch.diag(v)

        # Step 2: Similarity constraint adjustment
        # SimT_{i,k} = sum_j T_{i,j} * T_{k,j}
        SimT = T @ T.T  # (M, M)

        # Gradient: ∇T_{i,j} = 2λ * sum_k (Sim_{i,k} - SimT_{i,k}) * T_{k,j}
        diff = Sim - SimT  # (M, M)
        grad = 2 * lam * (diff @ T)  # (M, N)

        # Update T (gradient descent on similarity term)
        T = T - eta * grad

        # Step 3: Project back to feasible set
        T = T.clamp(min=0)
        # Re-normalize to satisfy marginals
        T = T / (T.sum(dim=1, keepdim=True) + 1e-10) * r.unsqueeze(1)

        # Update K for next iteration (keep the modified T)
        # Reconstruct u, v from T for next Sinkhorn step
        u = T.sum(dim=1) / (K @ (T.sum(dim=0) / (K.T @ T.sum(dim=1) + 1e-10)) + 1e-10)
        v = c / (K.T @ u + 1e-10)

    return T


def extract_pseudo_labels(
    T: torch.Tensor,             # (M, N) transport plan
    delta: float = 0.5,         # threshold
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pseudo-labels from OT plan (Eq. 11).
    P_δ = {(i, j) | T_{ij} >= δ}

    Returns:
        indices: (num_pseudo, 2) tensor of (molecule_idx, protein_idx) pairs
        scores: (num_pseudo,) confidence scores
    """
    # Normalize T to [0, 1] range for thresholding
    T_norm = T / (T.max() + 1e-10)

    mask = T_norm >= delta
    indices = torch.nonzero(mask)  # (num_pseudo, 2)
    scores = T_norm[mask]

    return indices, scores
