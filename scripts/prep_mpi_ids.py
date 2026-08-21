"""Extract the MPI molecule/protein entities from the rebuilt KG and build
ID/name maps needed for real embeddings.

Outputs (data/kg/):
  mpi_molecule_ids.npy  - global KG ids of drugs in drug_protein edges
  mpi_protein_ids.npy   - global KG ids of proteins in drug_protein edges
  mpi_drug_map.pt       - {kg_id: (drugbank_id, drug_name)}
  mpi_prot_map.pt       - {kg_id: (ncbi_gene_id, gene_symbol)}
"""
import torch, numpy as np, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

kg = torch.load("data/kg/kg_data.pt", weights_only=False)
id2entity = kg["id2entity"]; entity2id = kg["entity2id"]
rel2id = kg["relation2id"]; dp = rel2id["drug_protein"]

# unique molecules / proteins appearing in drug_protein edges
mol_ids, prot_ids = set(), set()
mol_set = set(kg["molecule_ids"]); prot_set = set(kg["protein_ids"])
for h, r, t in kg["triples"]:
    if r != dp:
        continue
    for e in (h, t):
        if e in mol_set: mol_ids.add(e)
        elif e in prot_set: prot_ids.add(e)
mol_ids = np.array(sorted(mol_ids)); prot_ids = np.array(sorted(prot_ids))
print(f"MPI molecules: {len(mol_ids)}, MPI proteins: {len(prot_ids)}", flush=True)

# names from PrimeKG (vectorized; no iterrows)
db_name, ncbi_sym = {}, {}
for ch in pd.read_csv("data/raw/primekg/kg.csv", chunksize=1000000, low_memory=False,
                      usecols=["x_id","x_type","x_name","y_id","y_type","y_name"]):
    for col in ("x","y"):
        d = ch[ch[f"{col}_type"] == "drug"]
        db_name.update(zip(d[f"{col}_id"].astype(str), d[f"{col}_name"].astype(str)))
        p = ch[ch[f"{col}_type"] == "gene/protein"]
        ncbi_sym.update(zip(p[f"{col}_id"].astype(str), p[f"{col}_name"].astype(str)))

drug_map, prot_map = {}, {}
for kid in mol_ids:
    dbid = id2entity[int(kid)].replace("DRUGBANK:", "")
    drug_map[int(kid)] = (dbid, db_name.get(dbid, ""))
for kid in prot_ids:
    ncbi = id2entity[int(kid)].replace("NCBI:", "")
    prot_map[int(kid)] = (ncbi, ncbi_sym.get(ncbi, ""))

np.save("data/kg/mpi_molecule_ids.npy", mol_ids.astype(np.int64))
np.save("data/kg/mpi_protein_ids.npy", prot_ids.astype(np.int64))
torch.save(drug_map, "data/kg/mpi_drug_map.pt")
torch.save(prot_map, "data/kg/mpi_prot_map.pt")
print("sample drugs:", list(drug_map.items())[:3], flush=True)
print("sample prots:", list(prot_map.items())[:3], flush=True)
print("DONE", flush=True)
