"""Molecule embeddings via PubChem FTP bulk files (no REST throttling).

1. DrugBank ID -> CID  from CID-Synonym-filtered.gz (DrugBank accessions appear
   as PubChem synonyms).
2. CID -> SMILES       from CID-SMILES.gz.
3. Morgan fingerprint (r=2, 2048) -> seeded 2048x512 projection, L2-normalized.

Drugs without a PubChem entry get a zero vector (like the original ~67% coverage).
Outputs: data/embeddings/{mol_embeddings.pt, mol_kg_ids.npy, drug_smiles_real.npy}
"""
import torch, numpy as np, gzip, time, sys
from pathlib import Path

OUT = Path("data/embeddings"); OUT.mkdir(parents=True, exist_ok=True)
SYN = "data/raw/pubchem/CID-Synonym-filtered.gz"
SMI = "data/raw/pubchem/CID-SMILES.gz"

mol_ids = np.load("data/kg/mpi_molecule_ids.npy")
drug_map = torch.load("data/kg/mpi_drug_map.pt", weights_only=False)
# DrugBank id (upper) -> list of kg ids (usually one)
db_of_kid = {int(k): v[0].upper() for k, v in drug_map.items()}
want_db = set(db_of_kid.values())
print(f"{len(mol_ids)} drugs, {len(want_db)} unique DrugBank ids", flush=True)

# ---- Stage 1: DrugBank id -> CID ----
t0 = time.time()
db_cid = {}
with gzip.open(SYN, "rt") as f:
    for i, line in enumerate(f):
        tab = line.find("\t")
        if tab < 0:
            continue
        syn = line[tab + 1:].strip().upper()
        if syn in want_db and syn not in db_cid:
            db_cid[syn] = int(line[:tab])
        if (i + 1) % 20_000_000 == 0:
            print(f"  syn {i/1e6:.0f}M lines, mapped {len(db_cid)}/{len(want_db)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
print(f"DrugBank->CID: {len(db_cid)}/{len(want_db)}", flush=True)

# ---- Stage 2: CID -> SMILES (only needed cids) ----
need_cids = set(db_cid.values())
cid_smiles = {}
with gzip.open(SMI, "rt") as f:
    for i, line in enumerate(f):
        tab = line.find("\t")
        if tab < 0:
            continue
        cid = int(line[:tab])
        if cid in need_cids:
            cid_smiles[cid] = line[tab + 1:].strip()
            if len(cid_smiles) == len(need_cids):
                break
        if (i + 1) % 20_000_000 == 0:
            print(f"  smiles {i/1e6:.0f}M lines, got {len(cid_smiles)}/{len(need_cids)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
print(f"CID->SMILES: {len(cid_smiles)}/{len(need_cids)}", flush=True)

# ---- Stage 3: Morgan -> 512 ----
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

rng = np.random.RandomState(0)
proj = (rng.randn(2048, 512).astype(np.float32)) / np.sqrt(2048)
fps = np.zeros((len(mol_ids), 2048), dtype=np.float32)
smiles_out, n_valid = [], 0
for i, kid in enumerate(mol_ids):
    db = db_of_kid[int(kid)]
    smi = cid_smiles.get(db_cid.get(db, -1), "")
    smiles_out.append(smi)
    if not smi:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32); DataStructs.ConvertToNumpyArray(bv, arr)
    fps[i] = arr; n_valid += 1

emb = torch.nn.functional.normalize(torch.tensor(fps @ proj, dtype=torch.float32), dim=-1)
torch.save(emb, OUT / "mol_embeddings.pt")
np.save(OUT / "mol_kg_ids.npy", mol_ids.astype(np.int64))
np.save(OUT / "drug_smiles_real.npy", np.array(smiles_out, dtype=object))
print(f"coverage: {n_valid}/{len(mol_ids)} ({100*n_valid/len(mol_ids):.1f}%); "
      f"mol_embeddings {tuple(emb.shape)}", flush=True)
print("DONE", flush=True)
