"""
Run KG training with pseudo-labels + link prediction evaluation.
Matches paper Table 3: train KG embeddings on augmented graph, evaluate Hits@1/3/5.

Higher-fidelity reproduction:
  * Type-aware negative sampling: MPI/pseudo edges are corrupted with random
    PROTEIN tails (aligned with the protein-ranking evaluation), other edges
    with random entities.
  * Self-adversarial negative sampling with K negatives (RotatE-style loss).
  * dim=256 sparse embeddings + SparseAdam.
  * Filtered ranking among the real protein candidate set.
"""
import torch, torch.nn.functional as F, numpy as np, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.score_model import ScoreFunction
from models.kg_embeddings import get_kg_model
from utils.sinkhorn import sinkhorn_knopp, extract_pseudo_labels

# ---- hyper-parameters (overridable via env) ----
DIM = int(os.environ.get("KG_DIM", 256))
EPOCHS = int(os.environ.get("KG_EPOCHS", 80))
BATCH = int(os.environ.get("KG_BATCH", 4096))
N_NEG = int(os.environ.get("KG_NNEG", 64))   # negatives per positive
GAMMA = float(os.environ.get("KG_GAMMA", 9.0))  # margin in self-adv logistic loss (model-dependent)
ADV_TEMP = 1.0       # self-adversarial temperature
ALPHA = 0.1          # pseudo-label loss weight
LR = 2e-3
N_EVAL = int(os.environ.get("KG_NEVAL", 2000))  # number of test queries to rank
CAND_MAX = int(os.environ.get("KG_CANDMAX", 50000))  # eval candidate-pool size
NEG_POOL = os.environ.get("KG_NEGPOOL", "full")  # "full"=all proteins, "mpi"=MPI proteins
MODELS = os.environ.get("KG_MODELS", "rotate,pairre,mure,toruse,complex_ff").split(",")
TAG = os.environ.get("KG_TAG", "all")           # output filename suffix


def self_adv_loss(pos_score, neg_score):
    """RotatE self-adversarial negative sampling loss.
    pos_score: (B,)  neg_score: (B, K).  Score = -distance (higher is better)."""
    weights = F.softmax(neg_score * ADV_TEMP, dim=1).detach()
    pos_loss = -F.logsigmoid(GAMMA + pos_score)
    neg_loss = -(weights * F.logsigmoid(-(GAMMA + neg_score))).sum(dim=1)
    return (pos_loss + neg_loss).mean()


def main():
    device = torch.device("cuda:0")

    # === Stage 1: Load data ===
    print("=== Loading data ===", flush=True)
    triples = torch.from_numpy(np.load("data/kg/triples.npy")).long()
    metadata = torch.load("data/kg/metadata.pt", weights_only=False)
    mol_kg_ids = np.load("data/embeddings/mol_kg_ids.npy")
    prot_kg_ids = np.load("data/embeddings/prot_kg_ids.npy")

    num_entities = metadata["num_entities"]
    num_relations = metadata["num_relations"] + 1  # +1 for pseudo_interaction
    pseudo_rel_id = metadata["num_relations"]

    mol_set = set(mol_kg_ids.tolist())
    prot_set = set(prot_kg_ids.tolist())

    # Identify MPI edges and normalize orientation to (mol, r, prot).
    th = triples[:, 0].numpy(); tr = triples[:, 1].numpy(); tt = triples[:, 2].numpy()
    is_mol_h = np.isin(th, mol_kg_ids); is_prot_t = np.isin(tt, prot_kg_ids)
    is_prot_h = np.isin(th, prot_kg_ids); is_mol_t = np.isin(tt, mol_kg_ids)
    mpi_fwd = is_mol_h & is_prot_t          # (mol, r, prot)
    mpi_rev = is_prot_h & is_mol_t          # (prot, r, mol) -> flip
    mpi_mask = mpi_fwd | mpi_rev

    norm = triples.clone()
    rev_idx = np.where(mpi_rev)[0]
    norm[rev_idx] = norm[rev_idx][:, [2, 1, 0]]  # flip head/tail so tail=protein

    mpi_idx = np.where(mpi_mask)[0]
    other_idx = np.where(~mpi_mask)[0]

    # Hold out test MPI edges.
    np.random.seed(42)
    perm = np.random.permutation(len(mpi_idx))
    n_test = min(10000, len(mpi_idx) // 5)
    test_sel = mpi_idx[perm[:n_test]]
    train_mpi_sel = mpi_idx[perm[n_test:]]

    train_other = norm[other_idx]
    train_mpi = norm[train_mpi_sel]
    test_triples = norm[test_sel]
    train_triples = torch.cat([train_other, train_mpi], dim=0)
    # boolean mask: does this train triple's tail need a PROTEIN negative?
    train_tail_is_prot = torch.cat([
        torch.zeros(len(train_other), dtype=torch.bool),
        torch.ones(len(train_mpi), dtype=torch.bool),
    ])

    # Protein candidate pool = unique proteins appearing in MPI edges.
    mpi_prot_pool = np.unique(norm[mpi_idx][:, 2].numpy())
    # Full protein entity set (for paper-aligned negatives + large candidate pool).
    full_prot_ids = np.load("data/kg/protein_ids.npy").astype(np.int64)
    # Eval candidate pool: all MPI proteins (guarantees the true tail is present)
    # plus a random sample of other proteins, up to CAND_MAX. Ranking against this
    # larger pool (instead of only the 3094 MPI proteins) matches the paper's
    # "rank candidate protein nodes" and removes the small-pool advantage.
    rng = np.random.RandomState(123)
    extra = max(0, CAND_MAX - len(mpi_prot_pool))
    if extra > 0:
        samp = rng.choice(full_prot_ids, size=min(extra, len(full_prot_ids)), replace=False)
        cand_np = np.union1d(mpi_prot_pool, samp)
    else:
        cand_np = mpi_prot_pool
    print(f"  Entities: {num_entities}, Relations: {num_relations}", flush=True)
    print(f"  Train triples: {len(train_triples)} (MPI {len(train_mpi)})", flush=True)
    print(f"  Test MPI triples: {len(test_triples)}", flush=True)
    print(f"  MPI protein pool: {len(mpi_prot_pool)} | full protein set: {len(full_prot_ids)}", flush=True)
    print(f"  Eval candidate pool: {len(cand_np)} (CAND_MAX={CAND_MAX}) | neg pool: {NEG_POOL}", flush=True)

    # Filtered eval: mol -> set of all known true proteins.
    mol_to_prots = {}
    all_mpi = norm[mpi_idx].numpy()
    for h, r, t in all_mpi:
        mol_to_prots.setdefault(int(h), set()).add(int(t))

    # === Stage 2: Generate pseudo-labels ===
    print("\n=== Generating pseudo-labels ===", flush=True)
    mol_embs = torch.load("data/embeddings/mol_embeddings.pt", weights_only=False)
    prot_embs = torch.load("data/embeddings/prot_embeddings.pt", weights_only=False)

    model_score = ScoreFunction(mol_dim=512, prot_dim=512).to(device)
    model_score.load_state_dict(torch.load("data/embeddings/score_model.pt", weights_only=False))
    model_score.eval()
    with torch.no_grad():
        S = model_score.score_matrix(mol_embs.to(device), prot_embs.to(device)).cpu()
    C = 1.0 - S
    T = sinkhorn_knopp(C, eps=0.05, max_iter=50)
    pseudo_pairs, pseudo_scores = extract_pseudo_labels(T, delta=0.3)
    pseudo_triples_list = []
    for pair in pseudo_pairs:
        pseudo_triples_list.append([int(mol_kg_ids[pair[0]]), pseudo_rel_id, int(prot_kg_ids[pair[1]])])
    pseudo_triples = torch.tensor(pseudo_triples_list, dtype=torch.long) if pseudo_triples_list else None
    print(f"  Pseudo-labels: {len(pseudo_triples_list)} triples", flush=True)

    # === Stage 3: Train KG embeddings ===
    print("\n=== Training KG Embeddings ===", flush=True)
    train_triples_dev = train_triples.to(device)
    train_tail_is_prot_dev = train_tail_is_prot.to(device)
    pseudo_triples_dev = pseudo_triples.to(device) if pseudo_triples is not None else None
    # Negative pool for MPI/pseudo edges: full protein set by default (decoupled
    # from the eval candidate pool), or just MPI proteins if requested.
    neg_pool_np = full_prot_ids if NEG_POOL == "full" else mpi_prot_pool
    neg_pool_dev = torch.tensor(neg_pool_np, dtype=torch.long, device=device)
    n_neg_pool = neg_pool_dev.size(0)
    cand_prots = torch.tensor(cand_np, dtype=torch.long, device=device)  # eval candidate set

    def sample_neg_tails(tail_is_prot, B):
        """(B, N_NEG) negative tails: proteins for MPI/pseudo edges, else any entity."""
        neg_any = torch.randint(0, num_entities, (B, N_NEG), device=device)
        neg_prot = neg_pool_dev[torch.randint(0, n_neg_pool, (B, N_NEG), device=device)]
        return torch.where(tail_is_prot.unsqueeze(1), neg_prot, neg_any)

    results_all = {}
    for model_name in MODELS:
        print(f"\n--- {model_name} ---", flush=True)
        kg_model = get_kg_model(model_name, num_entities, num_relations, dim=DIM, margin=GAMMA, sparse=True).to(device)
        optimizer = torch.optim.SparseAdam(kg_model.parameters(), lr=LR)
        n_train = train_triples_dev.size(0)
        n_pseudo = pseudo_triples_dev.size(0) if pseudo_triples_dev is not None else 0

        for epoch in range(EPOCHS):
            kg_model.train()
            perm = torch.randperm(n_train, device=device)
            epoch_loss = torch.zeros((), device=device); n_batches = 0
            for i in range(0, n_train, BATCH):
                idx = perm[i:i+BATCH]
                batch = train_triples_dev[idx]
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                B = idx.size(0)
                neg_t = sample_neg_tails(train_tail_is_prot_dev[idx], B)
                pos_score = kg_model(h, r, t)
                neg_score = kg_model(h.unsqueeze(1).expand(B, N_NEG),
                                     r.unsqueeze(1).expand(B, N_NEG), neg_t)
                loss = self_adv_loss(pos_score, neg_score)

                if pseudo_triples_dev is not None and n_pseudo > 0:
                    p_idx = torch.randint(0, n_pseudo, (min(B, n_pseudo),), device=device)
                    pb = pseudo_triples_dev[p_idx]; Bp = p_idx.size(0)
                    p_neg = neg_pool_dev[torch.randint(0, n_neg_pool, (Bp, N_NEG), device=device)]
                    p_pos = kg_model(pb[:, 0], pb[:, 1], pb[:, 2])
                    p_neg_s = kg_model(pb[:, 0].unsqueeze(1).expand(Bp, N_NEG),
                                       pb[:, 1].unsqueeze(1).expand(Bp, N_NEG), p_neg)
                    loss = loss + ALPHA * self_adv_loss(p_pos, p_neg_s)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.detach(); n_batches += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={(epoch_loss/n_batches).item():.4f}", flush=True)

        # === Evaluate: filtered link prediction, rank true protein among candidates ===
        kg_model.eval()
        n_cand = cand_prots.size(0)
        ranks = []
        n_eval = min(N_EVAL, test_triples.size(0))
        with torch.no_grad():
            for i in range(n_eval):
                h, r, t = test_triples[i].tolist()
                h_exp = torch.full((n_cand,), h, dtype=torch.long, device=device)
                r_exp = torch.full((n_cand,), r, dtype=torch.long, device=device)
                scores = kg_model(h_exp, r_exp, cand_prots)
                true_score = kg_model(torch.tensor([h], device=device),
                                      torch.tensor([r], device=device),
                                      torch.tensor([t], device=device))
                # filtered: mask other known true proteins for this molecule
                known = mol_to_prots.get(h, set()) - {t}
                if known:
                    mask = torch.isin(cand_prots, torch.tensor(list(known), device=device))
                    scores = scores.masked_fill(mask, float("-inf"))
                rank = (scores > true_score).sum().item() + 1
                ranks.append(rank)

        hits1 = np.mean([r <= 1 for r in ranks]) * 100
        hits3 = np.mean([r <= 3 for r in ranks]) * 100
        hits5 = np.mean([r <= 5 for r in ranks]) * 100
        mrr = float(np.mean([1.0 / r for r in ranks]))
        print(f"  Hits@1={hits1:.1f}%, Hits@3={hits3:.1f}%, Hits@5={hits5:.1f}%, MRR={mrr:.4f}", flush=True)
        results_all[model_name] = {"hits1": hits1, "hits3": hits3, "hits5": hits5, "mrr": mrr}
        torch.save(results_all, f"data/kg/link_pred_{TAG}.pt")
        # Save trained KG embedding weights (entity + relation tables) for reuse.
        ckpt_path = f"data/kg/kg_emb_{TAG}_{model_name}.pt"
        torch.save({
            "model_name": model_name,
            "state_dict": kg_model.state_dict(),
            "dim": DIM, "gamma": GAMMA, "n_neg": N_NEG, "epochs": EPOCHS,
            "num_entities": num_entities, "num_relations": num_relations,
            "metrics": results_all[model_name],
        }, ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}", flush=True)

    # === Summary ===
    print("\n" + "=" * 60, flush=True)
    print("KGOT Link Prediction Results (filtered, self-adv, dim=%d)" % DIM, flush=True)
    print("=" * 60, flush=True)
    print(f"{'Model':<12} {'Hits@1':<8} {'Hits@3':<8} {'Hits@5':<8} {'MRR':<8}", flush=True)
    print("-" * 44, flush=True)
    for name, res in results_all.items():
        print(f"{name:<12} {res['hits1']:<8.1f} {res['hits3']:<8.1f} {res['hits5']:<8.1f} {res['mrr']:<8.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
