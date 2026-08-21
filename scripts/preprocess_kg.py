"""Pre-convert KG from torch pickle to fast numpy format."""
import torch, numpy as np, time

print("Loading KG (557MB)...", flush=True)
t0 = time.time()
data = torch.load("data/kg/kg_data.pt", weights_only=False)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
print(f"Entities: {len(data['entity2id'])}, Triples: {len(data['triples'])}", flush=True)

print("Converting to numpy...", flush=True)
triples_np = np.array(data["triples"], dtype=np.int32)
np.save("data/kg/triples.npy", triples_np)
print(f"  triples.npy: {triples_np.shape} ({triples_np.nbytes/1e6:.0f} MB)", flush=True)

mol_ids = np.array(data["molecule_ids"], dtype=np.int32)
prot_ids = np.array(data["protein_ids"], dtype=np.int32)
np.save("data/kg/molecule_ids.npy", mol_ids)
np.save("data/kg/protein_ids.npy", prot_ids)
print(f"  mol_ids: {len(mol_ids)}, prot_ids: {len(prot_ids)}", flush=True)

metadata = {
    "num_entities": len(data["entity2id"]),
    "num_relations": len(data["relation2id"]),
    "relation2id": data["relation2id"],
}
torch.save(metadata, "data/kg/metadata.pt")
print("DONE", flush=True)
