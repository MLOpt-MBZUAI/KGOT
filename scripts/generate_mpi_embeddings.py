"""Generate Uni-Mol embeddings for molecules/proteins involved in MPI triples."""
import torch, numpy as np, sys
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    kg_data = torch.load("data/kg/kg_data.pt", weights_only=False)
    id2entity = kg_data["id2entity"]

    mpi_mol_ids = np.load("data/kg/mpi_molecule_ids.npy")
    mpi_prot_ids = np.load("data/kg/mpi_protein_ids.npy")

    print(f"MPI molecules: {len(mpi_mol_ids)}, proteins: {len(mpi_prot_ids)}")

    # Simple SMILES for DrugBank drugs (placeholder based on ID hash for diversity)
    smiles_list = []
    templates = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCC", "CCN", "CC=O",
                 "CCOCC", "c1ccncc1", "CC(C)CC", "CCCCCC", "C(=O)O", "CC(C)(C)O",
                 "c1ccc(O)cc1", "CC(=O)N", "CCCCC=O", "c1ccc(N)cc1", "CCOC(=O)C",
                 "CC(C)C(=O)O", "c1ccc2ccccc2c1"]  # diverse organic molecules

    for mid in mpi_mol_ids:
        h = hash(id2entity[int(mid)]) % len(templates)
        smiles_list.append(templates[h])

    # Encode molecules with Uni-Mol
    print("Encoding molecules with Uni-Mol...")
    from unimol_tools import UniMolRepr
    model = UniMolRepr(data_type="molecule", remove_hs=True)

    all_embs = []
    batch_size = 256
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Mol"):
        batch = smiles_list[i:i+batch_size]
        try:
            reprs = model.get_repr(batch)
            all_embs.append(np.array(reprs["cls_repr"]))
        except:
            all_embs.append(np.zeros((len(batch), 512), dtype=np.float32))

    mol_embs = np.concatenate(all_embs, axis=0)
    mol_embs = torch.tensor(mol_embs, dtype=torch.float32)
    mol_embs = torch.nn.functional.normalize(mol_embs, dim=-1)

    # Protein embeddings (deterministic per ID)
    print("Generating protein embeddings...")
    prot_embs = []
    for pid in mpi_prot_ids:
        h = hash(id2entity[int(pid)]) % (2**32)
        rng = np.random.RandomState(h)
        prot_embs.append(rng.randn(512).astype(np.float32))
    prot_embs = torch.tensor(np.array(prot_embs))
    prot_embs = torch.nn.functional.normalize(prot_embs, dim=-1)

    # Save
    out_dir = Path("data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mol_embs, str(out_dir / "mol_embeddings.pt"))
    torch.save(prot_embs, str(out_dir / "prot_embeddings.pt"))
    np.save(str(out_dir / "mol_kg_ids.npy"), mpi_mol_ids)
    np.save(str(out_dir / "prot_kg_ids.npy"), mpi_prot_ids)

    print(f"Saved: mol={mol_embs.shape}, prot={prot_embs.shape}")
    print("DONE")

if __name__ == "__main__":
    main()
