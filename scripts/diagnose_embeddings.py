"""Diagnose why score function can't distinguish pairs."""
import torch, numpy as np

mol_embs = torch.load("data/embeddings/mol_embeddings.pt", weights_only=False)
prot_embs = torch.load("data/embeddings/prot_embeddings.pt", weights_only=False)

print("=== Embedding Analysis ===")
print(f"mol: {mol_embs.shape}, prot: {prot_embs.shape}")

# Mol diversity
mol_sim = mol_embs[:200] @ mol_embs[:200].t()
mask = ~torch.eye(200, dtype=torch.bool)
print(f"\nMol-Mol cosine sim (200x200):")
print(f"  mean={mol_sim[mask].mean():.4f}, std={mol_sim[mask].std():.4f}")
print(f"  range=[{mol_sim[mask].min():.4f}, {mol_sim[mask].max():.4f}]")

# Prot diversity
prot_sim = prot_embs[:200] @ prot_embs[:200].t()
mask_p = ~torch.eye(200, dtype=torch.bool)
print(f"\nProt-Prot cosine sim (200x200):")
print(f"  mean={prot_sim[mask_p].mean():.4f}, std={prot_sim[mask_p].std():.4f}")
print(f"  range=[{prot_sim[mask_p].min():.4f}, {prot_sim[mask_p].max():.4f}]")

# Diagnosis
if mol_sim[mask].std() < 0.05:
    print("\n*** PROBLEM: Molecule embeddings have very low diversity ***")
    print("All molecules map to nearly the same point in embedding space.")
    print("This is because we used placeholder SMILES (CCO, CCCC, c1ccccc1...)")
    print("Uni-Mol correctly encodes them, but 20 template molecules = only 20 unique vectors,")
    print("and 6282 drugs are mapped to these 20 via hash -> massive duplication.")

    # Verify: how many unique embeddings do we actually have?
    unique = torch.unique(mol_embs, dim=0)
    print(f"\nUnique molecule embeddings: {unique.shape[0]} / {mol_embs.shape[0]}")
else:
    print("\nMolecule embeddings look diverse enough")

if prot_sim[mask_p].std() < 0.05:
    print("\n*** Protein embeddings also have low diversity ***")
else:
    print("\nProtein embeddings have reasonable diversity")
    print("(hash-based random projection gives good spread)")
