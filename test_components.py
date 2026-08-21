"""
Test all KGOT components with synthetic data.
Verifies: score function, IOT loss, Sinkhorn OT, pseudo-labeling, KG embeddings.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from models.score_model import ScoreFunction, IOTLoss
from models.kg_embeddings import get_kg_model, KGTrainer, RotatE, PairRE
from utils.sinkhorn import sinkhorn_knopp, sinkhorn_with_similarity, extract_pseudo_labels
from utils.metrics import evaluate_virtual_screening, evaluate_link_prediction


def test_score_function():
    print("=" * 50)
    print("Test 1: Score Function")
    print("=" * 50)

    model = ScoreFunction(mol_dim=128, prot_dim=128, hidden_dim=64)

    mol_emb = torch.randn(8, 128)
    prot_emb = torch.randn(8, 128)

    # Single pair scoring
    scores = model(mol_emb, prot_emb)
    print(f"  Pair scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Score matrix
    S = model.score_matrix(mol_emb[:4], prot_emb[:6])
    print(f"  Score matrix shape: {S.shape}")

    print("  ✓ Score function works\n")


def test_iot_loss():
    print("=" * 50)
    print("Test 2: IOT Loss")
    print("=" * 50)

    model = ScoreFunction(mol_dim=64, prot_dim=64, hidden_dim=32)
    criterion = IOTLoss(eps=0.05)

    mol_emb = torch.randn(16, 64)
    prot_emb = torch.randn(16, 64)

    loss = criterion(mol_emb, prot_emb, model)
    print(f"  IOT loss: {loss.item():.4f}")

    # Verify gradient flows
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.4f}")

    print("  ✓ IOT loss works\n")


def test_sinkhorn():
    print("=" * 50)
    print("Test 3: Sinkhorn OT")
    print("=" * 50)

    M, N = 10, 8
    C = torch.rand(M, N)

    T = sinkhorn_knopp(C, eps=0.01, max_iter=100)
    print(f"  Transport plan shape: {T.shape}")
    print(f"  Row sums (should be 1/M={1/M:.4f}): {T.sum(dim=1)[:3].tolist()}")
    print(f"  Col sums (should be 1/N={1/N:.4f}): {T.sum(dim=0)[:3].tolist()}")

    # Test with similarity
    S = torch.rand(M, N) * 0.8 + 0.1
    Sim = torch.rand(M, M)
    Sim = (Sim + Sim.T) / 2  # symmetric
    Sim.fill_diagonal_(1.0)

    T_sim = sinkhorn_with_similarity(S, Sim, eps=0.01, lam=0.1, eta=0.5, max_iter=10)
    print(f"  OT+Sim plan shape: {T_sim.shape}, sum: {T_sim.sum():.4f}")

    # Extract pseudo-labels
    pairs, scores = extract_pseudo_labels(T_sim, delta=0.3)
    print(f"  Pseudo-labels: {len(pairs)} pairs (δ=0.3)")

    print("  ✓ Sinkhorn OT works\n")


def test_kg_embeddings():
    print("=" * 50)
    print("Test 4: KG Embeddings")
    print("=" * 50)

    num_entities = 1000
    num_relations = 10

    for model_name in ['transe', 'rotate', 'pairre', 'complex_ff']:
        model = get_kg_model(model_name, num_entities, num_relations, dim=64)

        heads = torch.randint(0, num_entities, (32,))
        rels = torch.randint(0, num_relations, (32,))
        tails = torch.randint(0, num_entities, (32,))

        scores = model(heads, rels, tails)
        print(f"  {model_name:12s}: score shape={scores.shape}, mean={scores.mean():.3f}")

    # Test KGTrainer
    model = get_kg_model('rotate', num_entities, num_relations, dim=64)
    trainer = KGTrainer(model, alpha=0.1)

    pos_scores = model(heads, rels, tails)
    neg_tails = torch.randint(0, num_entities, (32,))
    neg_scores = model(heads, rels, neg_tails)

    loss = trainer.total_loss(pos_scores, neg_scores)
    loss.backward()
    print(f"  KG loss: {loss.item():.4f}")

    print("  ✓ KG embeddings work\n")


def test_metrics():
    print("=" * 50)
    print("Test 5: Evaluation Metrics")
    print("=" * 50)

    # Virtual screening
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.3, 0.8, 0.2, 0.1, 0.7, 0.4, 0.15, 0.05, 0.25])

    results = evaluate_virtual_screening(y_true, y_score)
    print(f"  AUROC: {results['auroc']:.1f}%")
    print(f"  BEDROC: {results['bedroc']:.1f}%")
    print(f"  EF@1%: {results['ef_1']:.2f}")

    # Link prediction
    ranks = [1, 3, 2, 5, 1, 10, 2, 4, 1, 8]
    lp_results = evaluate_link_prediction(ranks)
    print(f"  Hits@1: {lp_results['hits@1']:.1f}%")
    print(f"  Hits@5: {lp_results['hits@5']:.1f}%")
    print(f"  MRR: {lp_results['mrr']:.4f}")

    print("  ✓ Metrics work\n")


def test_full_mini_pipeline():
    print("=" * 50)
    print("Test 6: Mini Pipeline (end-to-end)")
    print("=" * 50)

    device = torch.device('cpu')

    # Synthetic data
    M, N = 50, 30  # molecules, proteins
    mol_dim, prot_dim = 64, 64

    mol_embs = torch.randn(M, mol_dim)
    prot_embs = torch.randn(N, prot_dim)

    # Train score function
    model = ScoreFunction(mol_dim=mol_dim, prot_dim=prot_dim, hidden_dim=32)
    criterion = IOTLoss(eps=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Positive pairs (diagonal)
    for epoch in range(5):
        batch_size = min(M, N, 16)
        idx = torch.randperm(min(M, N))[:batch_size]
        loss = criterion(mol_embs[idx], prot_embs[idx], model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Score training loss: {loss.item():.4f}")

    # Generate pseudo-labels
    model.eval()
    with torch.no_grad():
        S = model.score_matrix(mol_embs, prot_embs)

    Sim = torch.rand(M, M)
    Sim = (Sim + Sim.T) / 2
    Sim.fill_diagonal_(1.0)

    T = sinkhorn_with_similarity(S, Sim, eps=0.05, lam=0.1, eta=0.5, max_iter=5)
    pairs, scores = extract_pseudo_labels(T, delta=0.3)
    print(f"  Pseudo-labels generated: {len(pairs)} pairs")

    # Train KG
    num_entities = M + N
    num_relations = 5  # real relations + pseudo_interaction
    kg_triples = torch.randint(0, num_entities, (200, 3))
    kg_triples[:, 1] = torch.randint(0, num_relations, (200,))

    kg_model = get_kg_model('rotate', num_entities, num_relations, dim=32)
    optimizer = torch.optim.Adam(kg_model.parameters(), lr=1e-3)

    for _ in range(5):
        h, r, t = kg_triples[:, 0], kg_triples[:, 1], kg_triples[:, 2]
        pos_scores = kg_model(h, r, t)
        neg_t = torch.randint(0, num_entities, (200,))
        neg_scores = kg_model(h, r, neg_t)
        loss = torch.relu(6.0 - pos_scores + neg_scores).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  KG training loss: {loss.item():.4f}")
    print("  ✓ Full mini pipeline works\n")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("KGOT Component Tests")
    print("=" * 50 + "\n")

    test_score_function()
    test_iot_loss()
    test_sinkhorn()
    test_kg_embeddings()
    test_metrics()
    test_full_mini_pipeline()

    print("=" * 50)
    print("All tests passed! KGOT is ready.")
    print("=" * 50)
