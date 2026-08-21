"""Train score function with InfoNCE loss (more robust than IOT for initial training)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.score_model import ScoreFunction

def main():
    device = torch.device("cuda:0")

    # Load embeddings
    mol_embs = torch.load("data/embeddings/mol_embeddings.pt", weights_only=False)
    prot_embs = torch.load("data/embeddings/prot_embeddings.pt", weights_only=False)
    mol_kg_ids = np.load("data/embeddings/mol_kg_ids.npy")
    prot_kg_ids = np.load("data/embeddings/prot_kg_ids.npy")
    triples = np.load("data/kg/triples.npy")

    # Build pairs
    mol_id_to_local = {int(gid): i for i, gid in enumerate(mol_kg_ids)}
    prot_id_to_local = {int(gid): i for i, gid in enumerate(prot_kg_ids)}
    mol_set = set(mol_kg_ids.tolist())
    prot_set = set(prot_kg_ids.tolist())

    pairs = []
    for h, r, t in triples:
        if h in mol_set and t in prot_set:
            ml, pl = mol_id_to_local.get(h), prot_id_to_local.get(t)
            if ml is not None and pl is not None:
                pairs.append([ml, pl])
        elif t in mol_set and h in prot_set:
            ml, pl = mol_id_to_local.get(t), prot_id_to_local.get(h)
            if ml is not None and pl is not None:
                pairs.append([ml, pl])

    pairs = torch.tensor(pairs, dtype=torch.long)
    print(f"Train pairs: {len(pairs)}")

    # Train with InfoNCE
    model = ScoreFunction(mol_dim=512, prot_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size = 64
    temperature = 0.07
    epochs = 30

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(pairs))
        total_loss = 0
        n_batch = 0

        for i in range(0, len(pairs), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 8:
                continue

            mol_idx = pairs[idx, 0]
            prot_idx = pairs[idx, 1]
            B = len(mol_idx)

            batch_mol = mol_embs[mol_idx].to(device)
            batch_prot = prot_embs[prot_idx].to(device)

            S = model.score_matrix(batch_mol, batch_prot)
            logits = S / temperature
            labels = torch.arange(B, device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}", flush=True)

    # Evaluate diversity
    model.eval()
    with torch.no_grad():
        subset_mol = mol_embs[:200].to(device)
        subset_prot = prot_embs[:200].to(device)
        S = model.score_matrix(subset_mol, subset_prot)
        print(f"\nScore matrix (200x200):", flush=True)
        print(f"  range=[{S.min():.4f}, {S.max():.4f}]", flush=True)
        print(f"  std={S.std():.4f}", flush=True)
        print(f"  diag mean={S.diag().mean():.4f}", flush=True)
        mask = ~torch.eye(200, dtype=torch.bool, device=device)
        print(f"  off-diag mean={S[mask].mean():.4f}", flush=True)
        print(f"  SEPARATION = {S.diag().mean() - S[mask].mean():.4f}", flush=True)

    torch.save(model.state_dict(), "data/embeddings/score_model.pt")
    print("\nScore model saved!", flush=True)

if __name__ == "__main__":
    main()
