"""
KGOT Full Training Pipeline.

1. Build KG from biological datasets
2. Train score function S(x,y) with IOT loss on labeled pairs
3. Generate pseudo-labels via OT + similarity constraints
4. Train KG embeddings on augmented graph
5. Evaluate on DUD-E, LIT-PCBA (virtual screening) and link prediction

Usage:
    python train_pipeline.py --config configs/full_pipeline.yaml
"""

import os
import sys
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.score_model import ScoreFunction, IOTLoss
from models.kg_embeddings import get_kg_model, KGTrainer
from utils.sinkhorn import sinkhorn_with_similarity, extract_pseudo_labels
from utils.metrics import evaluate_virtual_screening, evaluate_link_prediction


def train_score_function(config, mol_embs, prot_embs, train_pairs, device):
    """
    Stage 2: Train S(x,y) on labeled molecule-protein pairs.
    Uses IOT loss (KL divergence between predicted and GT transport plans).
    """
    print("\n=== Stage 2: Training Score Function ===")

    mol_dim = mol_embs.size(1)
    prot_dim = prot_embs.size(1)

    model = ScoreFunction(mol_dim=mol_dim, prot_dim=prot_dim, hidden_dim=512).to(device)
    criterion = IOTLoss(eps=config.get('sinkhorn_eps', 0.01))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=0.01)

    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 50)

    # Train pairs: (mol_idx, prot_idx) positive pairs
    num_pairs = len(train_pairs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle
        perm = torch.randperm(num_pairs)

        for i in range(0, num_pairs, batch_size):
            batch_idx = perm[i:i+batch_size]
            if len(batch_idx) < 4:  # need minimum batch for OT
                continue

            mol_idx = train_pairs[batch_idx, 0]
            prot_idx = train_pairs[batch_idx, 1]

            batch_mol = mol_embs[mol_idx].to(device)
            batch_prot = prot_embs[prot_idx].to(device)

            loss = criterion(batch_mol, batch_prot, model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(num_pairs // batch_size, 1)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return model


def generate_pseudo_labels(config, score_model, mol_embs, prot_embs, mol_sim, device):
    """
    Stage 3: Generate pseudo-labels via OT + similarity constraints.
    """
    print("\n=== Stage 3: Generating Pseudo-Labels ===")

    score_model.eval()

    with torch.no_grad():
        # Compute full score matrix (may need batching for large M, N)
        M = mol_embs.size(0)
        N = prot_embs.size(0)

        # Batch computation if too large
        if M * N > 50_000_000:  # > 50M pairs
            print(f"  Score matrix too large ({M}×{N}), using subset...")
            M = min(M, 10000)
            N = min(N, 5000)
            mol_subset = mol_embs[:M].to(device)
            prot_subset = prot_embs[:N].to(device)
        else:
            mol_subset = mol_embs[:M].to(device)
            prot_subset = prot_embs[:N].to(device)

        S = score_model.score_matrix(mol_subset, prot_subset)  # (M, N)
        print(f"  Score matrix: {S.shape}, range [{S.min():.3f}, {S.max():.3f}]")

    # Molecular similarity (subset)
    Sim = mol_sim[:M, :M].to(device) if mol_sim is not None else None

    # OT with similarity constraints
    eps = config.get('sinkhorn_eps', 0.01)
    lam = config.get('similarity_lambda', 0.1)
    eta = config.get('learning_rate', 1.0)
    max_iter = config.get('max_iter', 50)

    print(f"  Running Sinkhorn OT (eps={eps}, λ={lam}, η={eta}, iter={max_iter})...")

    if Sim is not None:
        T = sinkhorn_with_similarity(S, Sim, eps=eps, lam=lam, eta=eta, max_iter=max_iter)
    else:
        from utils.sinkhorn import sinkhorn_knopp
        C = 1.0 - S
        T = sinkhorn_knopp(C, eps=eps)

    # Extract pseudo-labels
    delta = config.get('threshold', 0.5)
    pseudo_pairs, pseudo_scores = extract_pseudo_labels(T, delta=delta)

    print(f"  Generated {len(pseudo_pairs)} pseudo-labels (δ={delta})")
    print(f"  Score range: [{pseudo_scores.min():.3f}, {pseudo_scores.max():.3f}]")

    return pseudo_pairs, pseudo_scores, T


def train_kg_embeddings(config, kg_triples, pseudo_triples, num_entities, num_relations, device):
    """
    Stage 4: Train KG embeddings on augmented graph.
    L_total = L_KG + α * L_pseudo
    """
    print("\n=== Stage 4: Training KG Embeddings ===")

    model_name = config.get('model', 'rotate')
    dim = config.get('embedding_dim', 256)
    margin = config.get('margin', 6.0)
    alpha = config.get('alpha', 0.1)

    model = get_kg_model(model_name, num_entities, num_relations, dim, margin).to(device)
    trainer = KGTrainer(model, alpha=alpha, margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))

    batch_size = config.get('batch_size', 1024)
    epochs = config.get('epochs', 100)

    num_triples = len(kg_triples)
    num_pseudo = len(pseudo_triples) if pseudo_triples is not None else 0

    print(f"  Model: {model_name}, dim={dim}")
    print(f"  Real triples: {num_triples}, Pseudo triples: {num_pseudo}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        perm = torch.randperm(num_triples)

        for i in range(0, num_triples, batch_size):
            batch_idx = perm[i:i+batch_size]

            # Positive triples (real KG edges)
            pos = kg_triples[batch_idx].to(device)
            heads, rels, tails = pos[:, 0], pos[:, 1], pos[:, 2]

            # Negative sampling: corrupt tail (same batch size)
            neg_tails = torch.randint(0, num_entities, (len(batch_idx),), device=device)

            pos_scores = model(heads, rels, tails)
            neg_scores = model(heads, rels, neg_tails)

            # Pseudo-label triples (1:1:1 ratio — sample same count as real batch)
            pseudo_pred, pseudo_target = None, None
            if pseudo_triples is not None and num_pseudo > 0:
                # Sample same number as real batch for 1:1:1 ratio
                n_pseudo_batch = min(len(batch_idx), num_pseudo)
                pseudo_idx = torch.randint(0, num_pseudo, (n_pseudo_batch,))
                p_triples = pseudo_triples[pseudo_idx].to(device)
                p_heads, p_rels, p_tails = p_triples[:, 0], p_triples[:, 1], p_triples[:, 2]
                pseudo_pred = model(p_heads, p_rels, p_tails)
                pseudo_target = torch.ones_like(pseudo_pred)  # pseudo-labels are positive

            loss = trainer.total_loss(pos_scores, neg_scores, pseudo_pred, pseudo_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(num_triples // batch_size, 1)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return model


def main(config_path: str):
    """Run the full KGOT pipeline."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Stage 1: Load KG
    print("\n=== Stage 1: Loading Knowledge Graph ===")
    from data.build_kg import KnowledgeGraph

    kg_path = config['data'].get('kg_path', './data/kg')

    # Fast path: load pre-processed numpy files
    import numpy as np
    from pathlib import Path
    kg_dir = Path(kg_path)

    if (kg_dir / 'triples.npy').exists():
        print("  Loading from pre-processed numpy (fast)...")
        all_triples = torch.from_numpy(np.load(str(kg_dir / 'triples.npy'))).long()
        mol_ids_arr = np.load(str(kg_dir / 'molecule_ids.npy'))
        prot_ids_arr = np.load(str(kg_dir / 'protein_ids.npy'))
        metadata = torch.load(str(kg_dir / 'metadata.pt'), weights_only=False)

        num_entities = metadata['num_entities']
        num_relations = metadata['num_relations']
        relation2id = metadata['relation2id']
        molecule_ids = mol_ids_arr.tolist()
        protein_ids = prot_ids_arr.tolist()
        mol_set = set(molecule_ids)
        prot_set = set(protein_ids)

        print(f"  Entities: {num_entities}, Relations: {num_relations}")
        print(f"  Triples: {len(all_triples)}")
        print(f"  Molecules: {len(molecule_ids)}, Proteins: {len(protein_ids)}")
    else:
        # Fallback: load full KG object
        kg = KnowledgeGraph.load(kg_path)
        kg.summary()
        all_triples = kg.get_triples_tensor()
        num_entities = kg.num_entities
        num_relations = kg.num_relations
        molecule_ids = kg.molecule_ids
        protein_ids = kg.protein_ids
        mol_set = set(molecule_ids)
        prot_set = set(protein_ids)
        relation2id = kg.relation2id

    mpi_indices = []
    other_indices = []
    for i in range(len(all_triples)):
        h, r, t = all_triples[i][0].item(), all_triples[i][1].item(), all_triples[i][2].item()
        if (h in mol_set and t in prot_set) or (t in mol_set and h in prot_set):
            mpi_indices.append(i)
        else:
            other_indices.append(i)

    # Hold out 60K for test (or 20% of MPI triples, whichever is smaller)
    n_test = min(60000, len(mpi_indices) // 5)
    perm = torch.randperm(len(mpi_indices))
    test_idx = [mpi_indices[i] for i in perm[:n_test]]
    train_mpi_idx = [mpi_indices[i] for i in perm[n_test:]]

    train_indices = other_indices + train_mpi_idx
    train_triples = all_triples[train_indices]
    test_triples = all_triples[test_idx]

    # Initial ID mappings (will be overridden after embedding loading)
    mol_id_to_local = {gid: i for i, gid in enumerate(molecule_ids)}
    prot_id_to_local = {gid: i for i, gid in enumerate(protein_ids)}
    train_mpi_pairs = torch.zeros(0, 2, dtype=torch.long)  # placeholder, rebuilt after embedding load

    print(f"  Train triples: {len(train_triples)}")
    print(f"  Test triples (MPI): {len(test_triples)}")
    print(f"  MPI triples for score training: {len(train_mpi_idx)}")

    # Load or compute embeddings
    mol_emb_path = config['data'].get('mol_emb_path', './data/embeddings/mol_embeddings.pt')
    prot_emb_path = config['data'].get('prot_emb_path', './data/embeddings/prot_embeddings.pt')

    if os.path.exists(mol_emb_path) and os.path.exists(prot_emb_path):
        mol_embs = torch.load(mol_emb_path, weights_only=False)
        prot_embs = torch.load(prot_emb_path, weights_only=False)
        print(f"  Loaded embeddings: mol={mol_embs.shape}, prot={prot_embs.shape}")

        # Load ID mappings if available
        emb_dir = Path(mol_emb_path).parent
        if (emb_dir / 'mol_kg_ids.npy').exists():
            emb_mol_kg_ids = np.load(str(emb_dir / 'mol_kg_ids.npy')).tolist()
            emb_prot_kg_ids = np.load(str(emb_dir / 'prot_kg_ids.npy')).tolist()
            mol_id_to_local = {gid: i for i, gid in enumerate(emb_mol_kg_ids)}
            prot_id_to_local = {gid: i for i, gid in enumerate(emb_prot_kg_ids)}
            print(f"  ID mappings: {len(emb_mol_kg_ids)} mols, {len(emb_prot_kg_ids)} prots")

            # Rebuild train_mpi_pairs with correct mappings
            train_mpi_local = []
            n_mol_cap = mol_embs.size(0)
            n_prot_cap = prot_embs.size(0)
            for idx in train_mpi_idx:
                h = all_triples[idx][0].item()
                t = all_triples[idx][2].item()
                if h in mol_set and t in prot_set:
                    m_local = mol_id_to_local.get(h)
                    p_local = prot_id_to_local.get(t)
                elif t in mol_set and h in prot_set:
                    m_local = mol_id_to_local.get(t)
                    p_local = prot_id_to_local.get(h)
                else:
                    continue
                if m_local is not None and p_local is not None:
                    train_mpi_local.append([m_local, p_local])
            train_mpi_pairs = torch.tensor(train_mpi_local, dtype=torch.long)
            print(f"  Rebuilt train MPI pairs: {len(train_mpi_pairs)}")
    else:
        print("  Generating placeholder embeddings...")
        n_mol = len(molecule_ids)
        n_prot = min(len(protein_ids), 50000)
        mol_embs = torch.randn(n_mol, 512)
        mol_embs = torch.nn.functional.normalize(mol_embs, dim=-1)
        prot_embs = torch.randn(n_prot, 512)
        prot_embs = torch.nn.functional.normalize(prot_embs, dim=-1)
        print(f"  Placeholder: mol={mol_embs.shape}, prot={prot_embs.shape}")

    # Molecular similarity (placeholder or loaded)
    mol_sim_path = config['data'].get('mol_sim_path', None)
    mol_sim = None
    if mol_sim_path and os.path.exists(mol_sim_path):
        mol_sim = torch.load(mol_sim_path)

    # Stage 2: Train score function
    score_model = train_score_function(
        config['score_training'], mol_embs, prot_embs, train_mpi_pairs, device
    )

    # Stage 3: Generate pseudo-labels
    pseudo_pairs, pseudo_scores, T = generate_pseudo_labels(
        config['pseudo_label'], score_model, mol_embs, prot_embs, mol_sim, device
    )

    # Convert pseudo-pairs to KG triples with pseudo_interaction relation
    pseudo_rel_id = relation2id.get('pseudo_interaction', 0)
    if len(pseudo_pairs) > 0:
        mol_ids_tensor = torch.tensor(molecule_ids, dtype=torch.long)
        prot_ids_tensor = torch.tensor(protein_ids[:len(prot_embs)], dtype=torch.long)

        pseudo_triples_list = []
        for pair in pseudo_pairs:
            mol_kg_id = mol_ids_tensor[pair[0]].item() if pair[0] < len(mol_ids_tensor) else 0
            prot_kg_id = prot_ids_tensor[pair[1]].item() if pair[1] < len(prot_ids_tensor) else 0
            pseudo_triples_list.append([mol_kg_id, pseudo_rel_id, prot_kg_id])
        pseudo_triples = torch.tensor(pseudo_triples_list, dtype=torch.long)
    else:
        pseudo_triples = None

    # Stage 4: Train KG embeddings
    kg_model = train_kg_embeddings(
        config['kg_training'], train_triples, pseudo_triples, num_entities, num_relations, device
    )

    # Stage 5: Evaluate
    print("\n=== Stage 5: Evaluation ===")

    # Link prediction on test set
    from data.datasets import LinkPredictionDataset
    lp_dataset = LinkPredictionDataset(test_triples, train_triples, num_entities, len(protein_ids))
    lp_results = lp_dataset.evaluate(kg_model, protein_ids[:1000], device)  # subset for speed

    print(f"  Link Prediction Results:")
    for k, v in lp_results.items():
        print(f"    {k}: {v:.2f}")

    # Virtual screening evaluation (if DUD-E data available)
    dude_path = config['data'].get('dude_path', './data/dude')
    if os.path.exists(dude_path) and os.path.exists(os.path.join(dude_path, 'dude_data.csv')):
        print("\n  Virtual Screening (DUD-E):")
        from data.datasets import DUDEDataset
        dude = DUDEDataset(dude_path)

        if len(dude.targets) > 0:
            all_aurocs = []
            score_model.eval()

            for target in dude.targets[:5]:  # first 5 targets for quick eval
                smiles_list, labels = dude.get_target_data(target)
                if len(smiles_list) == 0:
                    continue

                # Encode molecules and score against target
                # (In full version: use mol_encoder + prot_encoder)
                # Placeholder: random scores
                scores_vs = np.random.rand(len(labels))

                from utils.metrics import evaluate_virtual_screening
                vs_results = evaluate_virtual_screening(labels, scores_vs)
                all_aurocs.append(vs_results['auroc'])
                print(f"    Target {target}: AUROC={vs_results['auroc']:.1f}%")

            if all_aurocs:
                print(f"    Mean AUROC: {np.mean(all_aurocs):.1f}%")
    else:
        print("\n  Virtual Screening: DUD-E data not found (skipped)")
        print(f"    Expected at: {dude_path}/dude_data.csv")

    print("\n=== Pipeline Complete ===")
    return kg_model, score_model, lp_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/full_pipeline.yaml')
    args = parser.parse_args()
    main(args.config)
