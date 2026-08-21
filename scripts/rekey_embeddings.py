"""Re-key mol/prot embeddings to the reconciled KG's new global ids.
Embedding vectors are unchanged; only mol_kg_ids.npy / prot_kg_ids.npy are updated
so each embedding row points to the correct entity in the rebuilt KG.
"""
import torch, numpy as np, gzip
kg = torch.load("data/kg/kg_data.pt", weights_only=False)
e2id = kg["entity2id"]

gene2uni = {}
with gzip.open("data/raw/idmap/HUMAN_9606_idmapping.dat.gz", "rt") as f:
    for line in f:
        acc, typ, val = line.rstrip("\n").split("\t")
        if typ == "GeneID" and val not in gene2uni:
            gene2uni[val] = acc

drug_map = torch.load("data/kg/mpi_drug_map.pt", weights_only=False)
prot_map = torch.load("data/kg/mpi_prot_map.pt", weights_only=False)

# Use the STABLE original MPI id lists (embedding rows are in this order) as the
# identity source, not the mutable *_kg_ids.npy which get overwritten each re-key.
old_mol = np.load("data/kg/mpi_molecule_ids.npy")
old_prot = np.load("data/kg/mpi_protein_ids.npy")

new_mol, miss_m = [], 0
for oid in old_mol:
    dbid = drug_map[int(oid)][0]
    ent = f"DRUGBANK:{dbid}"
    new_mol.append(e2id.get(ent, -1)); miss_m += ent not in e2id

new_prot, miss_p, recon = [], 0, 0
for oid in old_prot:
    gene = prot_map[int(oid)][0]
    acc = gene2uni.get(gene)
    ent = f"UNIPROT:{acc}" if acc else f"NCBI:{gene}"
    recon += acc is not None
    new_prot.append(e2id.get(ent, -1)); miss_p += ent not in e2id

new_mol = np.array(new_mol, dtype=np.int64)
new_prot = np.array(new_prot, dtype=np.int64)
np.save("data/embeddings/mol_kg_ids.npy", new_mol)
np.save("data/embeddings/prot_kg_ids.npy", new_prot)
print(f"mol re-keyed: {len(new_mol)} (missing {miss_m})", flush=True)
print(f"prot re-keyed: {len(new_prot)} (missing {miss_p}, reconciled {recon})", flush=True)
print("DONE", flush=True)
