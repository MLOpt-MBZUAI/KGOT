"""
Knowledge Graph Embedding Models for Link Prediction.

Supports: PairRE, RotatE, MuRE, TorusE, ComplEx-FF
All trained on augmented KG (real edges + pseudo interaction relation).

Training loss: L_total = L_KG + α * L_pseudo (Eq. 14)

Reference: Section 2.4, Table 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TransE(nn.Module):
    """TransE: h + r ≈ t"""
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_emb = nn.Embedding(num_relations, dim, sparse=sparse)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h, r, t):
        """Score function: -||h + r - t||"""
        return -torch.norm(h + r - t, dim=-1)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return self.score(h, r, t)


class RotatE(nn.Module):
    """RotatE: h ∘ r ≈ t (complex rotation)"""
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.dim = dim
        self.entity_emb = nn.Embedding(num_entities, dim * 2, sparse=sparse)  # real + imag
        self.relation_emb = nn.Embedding(num_relations, dim, sparse=sparse)   # phase
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.uniform_(self.relation_emb.weight, -torch.pi, torch.pi)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        t = self.entity_emb(tails)
        r_phase = self.relation_emb(relations)

        h_re, h_im = h.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # Rotation: (h_re + i*h_im) * (r_re + i*r_im)
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        score = -torch.norm(torch.cat([rot_re - t_re, rot_im - t_im], dim=-1), dim=-1)
        return score


class PairRE(nn.Module):
    """PairRE: paired relation vectors for head and tail."""
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_head = nn.Embedding(num_relations, dim, sparse=sparse)
        self.relation_tail = nn.Embedding(num_relations, dim, sparse=sparse)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_head.weight)
        nn.init.xavier_uniform_(self.relation_tail.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        t = self.entity_emb(tails)
        r_h = self.relation_head(relations)
        r_t = self.relation_tail(relations)

        score = -torch.norm(h * r_h - t * r_t, dim=-1)
        return score


class ComplExFF(nn.Module):
    """ComplEx with feed-forward scoring."""
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim * 2, sparse=sparse)
        self.relation_emb = nn.Embedding(num_relations, dim * 2, sparse=sparse)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = r.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)

        score = (h_re * r_re * t_re + h_im * r_re * t_im +
                 h_re * r_im * t_im - h_im * r_im * t_re).sum(dim=-1)
        return score


class MuRE(nn.Module):
    """
    MuRE: Multi-relational Poincaré graph embeddings.
    Score: -||R_r * h + r - t||^2 + b_h + b_t
    where R_r is a diagonal relation matrix, r is translation.
    """
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_diag = nn.Embedding(num_relations, dim, sparse=sparse)  # diagonal R_r
        self.relation_trans = nn.Embedding(num_relations, dim, sparse=sparse)  # translation r
        self.entity_bias = nn.Embedding(num_entities, 1, sparse=sparse)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.ones_(self.relation_diag.weight)
        nn.init.xavier_uniform_(self.relation_trans.weight)
        nn.init.zeros_(self.entity_bias.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        t = self.entity_emb(tails)
        R = self.relation_diag(relations)
        r = self.relation_trans(relations)
        b_h = self.entity_bias(heads).squeeze(-1)
        b_t = self.entity_bias(tails).squeeze(-1)

        score = -torch.norm(R * h + r - t, dim=-1) ** 2 + b_h + b_t
        return score


class TorusE(nn.Module):
    """
    TorusE: Knowledge graph embedding on a Lie group (torus).
    Embeddings live on [0, 1)^d (torus). Distance is computed modulo 1.
    Score: -||[h + r - t] mod 1||
    where mod 1 maps to [-0.5, 0.5) for distance computation.
    """
    def __init__(self, num_entities, num_relations, dim=256, margin=6.0, sparse=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim, sparse=sparse)
        self.relation_emb = nn.Embedding(num_relations, dim, sparse=sparse)
        self.margin = margin
        # Initialize in [0, 1)
        nn.init.uniform_(self.entity_emb.weight, 0.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, 0.0, 1.0)

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        # Compute h + r - t, then map to torus [-0.5, 0.5)
        diff = h + r - t
        # Modulo operation: map to [-0.5, 0.5)
        diff = diff - torch.floor(diff + 0.5)

        score = -torch.norm(diff, dim=-1)
        return score


def get_kg_model(name: str, num_entities: int, num_relations: int, dim: int = 256, margin: float = 6.0, sparse: bool = False):
    """Factory for KG embedding models."""
    models = {
        'transe': TransE,
        'rotate': RotatE,
        'pairre': PairRE,
        'complex_ff': ComplExFF,
        'mure': MuRE,
        'toruse': TorusE,
    }
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name.lower()](num_entities, num_relations, dim, margin, sparse=sparse)


class KGTrainer:
    """
    Training loop for KG embeddings with pseudo-label augmentation.

    L_total = L_KG + α * L_pseudo

    where L_KG is margin-based ranking loss on real triples,
    and L_pseudo aligns predicted scores with pseudo-label scores T.
    """

    def __init__(self, model, alpha: float = 0.1, margin: float = 6.0):
        self.model = model
        self.alpha = alpha
        self.margin = margin

    def kg_loss(self, pos_scores, neg_scores):
        """Margin ranking loss."""
        return F.relu(self.margin - pos_scores + neg_scores).mean()

    def pseudo_loss(self, pred_scores, target_scores):
        """L_pseudo = MSE between KG-predicted scores and pseudo-label scores T."""
        return F.mse_loss(pred_scores, target_scores)

    def total_loss(self, pos_scores, neg_scores, pseudo_pred=None, pseudo_target=None):
        """L_total = L_KG + α * L_pseudo"""
        loss = self.kg_loss(pos_scores, neg_scores)

        if pseudo_pred is not None and pseudo_target is not None:
            loss = loss + self.alpha * self.pseudo_loss(pseudo_pred, pseudo_target)

        return loss
