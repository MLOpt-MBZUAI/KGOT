"""Regenerate molecule embeddings using real SMILES + Uni-Mol."""
import torch, numpy as np
from tqdm import tqdm
from unimol_tools import UniMolRepr

# Load real SMILES
smiles_all = np.load("data/embeddings/drug_smiles_real.npy", allow_pickle=True)
print(f"Total drugs: {len(smiles_all)}")
print(f"With real SMILES: {sum(1 for s in smiles_all if s is not None)}")

# For drugs without SMILES, use "C" as minimal placeholder
smiles_list = [s if s is not None else "C" for s in smiles_all]

# Encode with Uni-Mol
print("Encoding with Uni-Mol...")
model = UniMolRepr(data_type="molecule", remove_hs=True)

all_embs = []
batch_size = 128
for i in tqdm(range(0, len(smiles_list), batch_size), desc="Uni-Mol"):
    batch = smiles_list[i:i+batch_size]
    try:
        reprs = model.get_repr(batch)
        all_embs.append(np.array(reprs["cls_repr"]))
    except:
        all_embs.append(np.zeros((len(batch), 512), dtype=np.float32))

embeddings = np.concatenate(all_embs, axis=0)
embeddings = torch.tensor(embeddings, dtype=torch.float32)
embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

# Verify diversity
sim = embeddings[:200] @ embeddings[:200].t()
mask = ~torch.eye(200, dtype=torch.bool)
print(f"\nEmbedding diversity check (200x200):")
print(f"  cosine sim: mean={sim[mask].mean():.4f}, std={sim[mask].std():.4f}")
print(f"  unique vectors: {torch.unique(embeddings, dim=0).shape[0]} / {embeddings.shape[0]}")

torch.save(embeddings, "data/embeddings/mol_embeddings.pt")
print(f"\nSaved: {embeddings.shape} → data/embeddings/mol_embeddings.pt")
