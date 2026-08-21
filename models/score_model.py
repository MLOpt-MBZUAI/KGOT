"""
Molecule-Protein Interaction Scoring Function.

S(x, y) = W(f(x) ⊕ g(y))

where f(x) = Uni-Mol molecule encoder, g(y) = Uni-Mol protein encoder,
W = trainable MLP, ⊕ = concatenation.

Training uses inverse optimal transport (IOT) loss:
L_score = KL(T_pred || T_gt)

Reference: Section 2.3, Eq. 2-5 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScoreFunction(nn.Module):
    """
    Interaction score function S(x, y) = W(f(x) ⊕ g(y)).

    Takes pre-computed molecule and protein embeddings and predicts
    interaction strength.
    """

    def __init__(self, mol_dim: int = 512, prot_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        # Paper Appendix C: "two layers of size [512, 256, 1], with ReLU activations"
        self.score_net = nn.Sequential(
            nn.Linear(mol_dim + prot_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, mol_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_emb: (B, mol_dim) molecule embeddings from Uni-Mol
            prot_emb: (B, prot_dim) protein embeddings from Uni-Mol

        Returns:
            scores: (B,) interaction scores in [0, 1]
        """
        combined = torch.cat([mol_emb, prot_emb], dim=-1)
        return self.score_net(combined).squeeze(-1)

    def score_matrix(self, mol_embs: torch.Tensor, prot_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute all-pairs score matrix S ∈ R^{M×N}.
        Uses batched computation to avoid OOM.
        """
        M = mol_embs.size(0)
        N = prot_embs.size(0)

        # Batch over molecules to avoid OOM
        batch_size = min(256, M)
        S = torch.zeros(M, N, device=mol_embs.device)

        for i in range(0, M, batch_size):
            mol_batch = mol_embs[i:i+batch_size]  # (bs, d)
            bs = mol_batch.size(0)
            mol_exp = mol_batch.unsqueeze(1).expand(bs, N, -1)
            prot_exp = prot_embs.unsqueeze(0).expand(bs, N, -1)
            combined = torch.cat([mol_exp, prot_exp], dim=-1)
            S[i:i+bs] = self.score_net(combined).squeeze(-1)

        return S


class IOTLoss(nn.Module):
    """
    Inverse Optimal Transport Loss for training the score function.

    Given a batch of positive pairs (x_i, y_i), constructs ground truth
    cost matrix C_gt and computes KL divergence between predicted and
    ground truth transport plans.

    L_score = KL(T_pred || T_gt) (Eq. 5)

    Reference: Section 2.3, Eq. 3-5.
    """

    def __init__(self, eps: float = 0.01):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mol_embs: torch.Tensor,    # (N, mol_dim) batch of molecule embeddings
        prot_embs: torch.Tensor,   # (N, prot_dim) batch of protein embeddings
        score_fn: ScoreFunction,   # the scoring model
    ) -> torch.Tensor:
        """
        Compute IOT loss on a batch of positive pairs.

        In each batch, (mol_embs[i], prot_embs[i]) is a positive pair.
        All other combinations are negatives.
        """
        N = mol_embs.size(0)
        device = mol_embs.device

        # Compute predicted score matrix for this batch
        S_pred = score_fn.score_matrix(mol_embs, prot_embs)  # (N, N)

        # Predicted cost and transport
        C_pred = 1.0 - S_pred
        T_pred = self._sinkhorn(C_pred)

        # Ground truth cost: C_gt[i,j] = 0 if j==i (positive), 1 otherwise
        C_gt = torch.ones(N, N, device=device)
        C_gt.fill_diagonal_(0)
        T_gt = self._sinkhorn(C_gt)

        # KL divergence: KL(T_pred || T_gt)
        T_pred_log = torch.log(T_pred + 1e-10)
        T_gt_log = torch.log(T_gt + 1e-10)
        kl = (T_pred * (T_pred_log - T_gt_log)).sum()

        return kl

    def _sinkhorn(self, C: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
        """Fast Sinkhorn for small batch matrices."""
        N = C.size(0)
        device = C.device

        r = torch.ones(N, device=device) / N
        c = torch.ones(N, device=device) / N

        K = torch.exp(-C / self.eps)
        u = torch.ones(N, device=device)
        v = torch.ones(N, device=device)

        for _ in range(max_iter):
            u = r / (K @ v + 1e-10)
            v = c / (K.T @ u + 1e-10)

        T = torch.diag(u) @ K @ torch.diag(v)
        return T
